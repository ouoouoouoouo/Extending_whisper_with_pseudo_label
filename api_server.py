# api_server.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import (
    WhisperForConditionalGeneration, 
    WhisperTokenizer, 
    WhisperProcessor,
    WhisperFeatureExtractor
)
import torchaudio
import torchaudio.functional
import subprocess
import tempfile
import soundfile as sf
import io
from typing import List, Dict
import os

GPU_ID = 3
PORT = 8001

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

app = FastAPI(title="Speech Emotion Recognition API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "/home/ouo/whisper_emotion/workspace/whisper_emotion_bf16_beta_scan/beta_0.30"
model = None
tokenizer = None
processor = None

# Token 映射
SLE_TOKENS = {
    51865: "happy",
    51866: "sad", 
    51867: "angry",
    51868: "neutral"
}

WLE_TOKENS = {
    51869: "happy",
    51870: "sad",
    51871: "angry", 
    51872: "neutral"
}

@app.on_event("startup")
async def load_model():
    """啟動時載入模型"""
    global model, tokenizer, processor
    
    print(f"Loading model from: {MODEL_PATH}")
    
    try:
        print("Loading fine-tuned model...")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
        model.eval()
        
        print("Loading custom tokenizer...")
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH)
        
        print("Loading feature extractor from base model...")
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v2")
        
        print("Creating processor...")
        processor = WhisperProcessor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )
        
        if torch.cuda.is_available():
            model.cuda()
            print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("✓ Model loaded on CPU")
            
        print("✓ All components loaded successfully!")
            
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
def root():
    return {
        "message": "Speech Emotion Recognition API",
        "version": "1.0",
        "status": "running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_emotion(audio_file: UploadFile = File(...), language: str = "zh"):
    """
    上傳音檔進行情感辨識
    
    支援格式：WAV, MP3, FLAC, M4A, OGG, AAC 等所有 ffmpeg 支援的格式
    回傳：轉錄文字 + 逐字情感標註
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # ========== 統一音訊處理：全部用 ffmpeg 轉成 16kHz mono WAV ==========
        audio_bytes = await audio_file.read()
        
        print(f"Processing audio file: {audio_file.filename}")
        
        # 獲取原始檔案副檔名
        file_ext = audio_file.filename.split('.')[-1].lower() if '.' in audio_file.filename else 'raw'
        
        # 創建臨時檔案
        with tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
            tmp_out_path = tmp_out.name
        
        try:
            # 用 ffmpeg 轉成 16kHz mono WAV
            print(f"Converting {file_ext} to 16kHz mono WAV...")
            result = subprocess.run([
                'ffmpeg',
                '-loglevel', 'error',  # 只顯示錯誤
                '-i', tmp_in_path,
                '-ar', '16000',
                '-ac', '1',
                '-sample_fmt', 's16',
                '-y',
                tmp_out_path
            ], check=True, capture_output=True)
            
            print("✓ Conversion successful")
            
            # 讀取轉換後的 WAV
            waveform, sr = sf.read(tmp_out_path, dtype='float32')
            waveform = torch.from_numpy(waveform)
            
            # 確保是 2D tensor [1, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            # 驗證採樣率
            if sr != 16000:
                print(f"Warning: Expected 16kHz but got {sr}Hz, resampling...")
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.decode() if e.stderr else str(e)
            raise HTTPException(
                status_code=400,
                detail=f"音檔轉換失敗 ({file_ext}): {error_msg}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"無法讀取音檔 '{audio_file.filename}': {str(e)}"
            )
        finally:
            # 清理臨時檔案
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)
        
        # 準備輸入特徵
        audio_array = waveform.squeeze().numpy()
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        if torch.cuda.is_available():
            input_features = input_features.cuda()
        
        # ========== 語言處理（不變）==========
        sot_id = tokenizer.bos_token_id
        transcribe_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")

        print(f"✓ Using language: {language}")
        lang_token = f"<|{language}|>"
        lang_id = tokenizer.convert_tokens_to_ids(lang_token)
        
        if lang_id == tokenizer.unk_token_id:
            raise HTTPException(
                status_code=400,
                detail=f"不支援的語言代碼: {language}。常用語言：zh(中文), en(英文), fr(法文), ja(日文)"
            )
        
        # ========== 第一階段：預測 SLE（不變）==========
        decoder_input_for_sle = torch.tensor(
            [[sot_id, lang_id, transcribe_id]],
            dtype=torch.long,
            device=input_features.device
        )
        
        sle_id_list = list(SLE_TOKENS.keys())
        vocab_size = len(tokenizer)
        sle_mask = torch.full((vocab_size,), float('-inf'), device=input_features.device)
        for token_id in sle_id_list:
            if 0 <= token_id < vocab_size:
                sle_mask[token_id] = 0.0
        
        with torch.no_grad():
            outputs = model(
                input_features=input_features,
                decoder_input_ids=decoder_input_for_sle,
                labels=None
            )
            next_token_logits = outputs.logits[:, -1, :]
            masked_logits = next_token_logits + sle_mask
            pred_sle_id = torch.argmax(masked_logits, dim=-1).item()
            probs = torch.softmax(masked_logits, dim=-1)
            confidence = probs[0, pred_sle_id].item()
        
        sentence_emotion = SLE_TOKENS.get(pred_sle_id, 'unknown')
        print(f"✓ Predicted SLE: {sentence_emotion} (ID: {pred_sle_id}, Confidence: {confidence:.2%})")
        
        # ========== 第二階段：生成轉錄（不變）==========
        decoder_input_for_asr = torch.tensor(
            [[sot_id, lang_id, transcribe_id, pred_sle_id]],
            dtype=torch.long,
            device=input_features.device
        )
        
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features=input_features,
                decoder_input_ids=decoder_input_for_asr,
                max_length=448,
                min_length=5,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
        
        print(f"✓ Generated {len(predicted_ids[0])} tokens")
        print(f"Token IDs (前 50): {predicted_ids[0].tolist()[:50]}")
        
        generated_ids_no_prompt = predicted_ids[:, 4:]
        
        transcription_with_tokens = tokenizer.decode(
            generated_ids_no_prompt[0],
            skip_special_tokens=False
        )
        
        text_only = tokenizer.decode(
            generated_ids_no_prompt[0],
            skip_special_tokens=True
        )
        
        word_emotions = extract_word_emotions(generated_ids_no_prompt[0].tolist())
        
        emotion_stats = calculate_emotion_stats(word_emotions)
        emotion_stats["sentence_emotion"] = sentence_emotion
        emotion_stats["sentence_confidence"] = confidence
        
        return {
            "success": True,
            "filename": audio_file.filename,
            "language": language,
            "sentence_emotion": sentence_emotion,
            "sentence_confidence": confidence,
            "text": text_only,
            "word_emotions": word_emotions,
            "emotion_statistics": emotion_stats,
            "raw_output": transcription_with_tokens
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"處理錯誤: {str(e)}")
        


def extract_word_emotions(token_ids: List[int]) -> List[Dict]:
    """從 token IDs 提取 word-level emotions"""
    words = []
    current_emotion = None
    current_text = ""
    
    for token_id in token_ids:
        if token_id in SLE_TOKENS:
            current_emotion = SLE_TOKENS[token_id]
            continue
        
        if token_id in WLE_TOKENS:
            if current_text.strip():
                words.append({
                    "text": current_text.strip(),
                    "emotion": current_emotion or "unknown"
                })
            current_text = ""
            current_emotion = WLE_TOKENS[token_id]
            continue
        
        if token_id >= 50257:
            continue
        
        word = tokenizer.decode([token_id])
        current_text += word
    
    if current_text.strip():
        words.append({
            "text": current_text.strip(),
            "emotion": current_emotion or "unknown"
        })
    
    return words


def calculate_emotion_stats(word_emotions: List[Dict]) -> Dict:
    """計算情感分布統計"""
    if not word_emotions:
        return {}
    
    emotion_counts = {"happy": 0, "sad": 0, "angry": 0, "neutral": 0, "unknown": 0}
    
    for item in word_emotions:
        emotion = item.get("emotion", "unknown")
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
    
    total = len(word_emotions)
    emotion_percentages = {
        emotion: round(count / total * 100, 2)
        for emotion, count in emotion_counts.items()
    }
    
    return {
        "total_words": total,
        "counts": emotion_counts,
        "percentages": emotion_percentages
    }


@app.post("/predict_simple")
async def predict_simple(audio_file: UploadFile = File(...)):
    """簡化版：只回傳文字和整體情感傾向"""
    result = await predict_emotion(audio_file)
    
    stats = result["emotion_statistics"]["percentages"]
    dominant_emotion = max(stats.items(), key=lambda x: x[1])
    
    return {
        "text": result["text"],
        "sentence_emotion": result["sentence_emotion"],
        "dominant_emotion": dominant_emotion[0],
        "emotion_percentage": dominant_emotion[1]
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting API on GPU {GPU_ID}, Port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
