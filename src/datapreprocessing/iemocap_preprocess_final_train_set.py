"""
IEMOCAP Dataset Preprocessing with emotion2vec
使用預訓練的 emotion2vec 模型生成 frame-level 情緒標籤
"""

import os
import json
import torch
import torchaudio
import whisper
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import re
from transformers import AutoModel, AutoTokenizer



# 設置 audio backend
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass

class IEMOCAPPreprocessor:
    """
    使用 emotion2vec 預處理 IEMOCAP
    """
    
    def __init__(self, iemocap_path, output_path):
        self.iemocap_path = Path(iemocap_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Emotion mapping
        self.emotion_map = {
            'neu': 'neutral',
            'hap': 'happy',
            'exc': 'happy',
            'sad': 'sad',
            'ang': 'angry',
            'fru': 'angry',
            'fea': 'neutral',
            'sur': 'neutral',
            'dis': 'neutral',
            'oth': 'neutral'
        }
        
        # emotion2vec 支持的情緒到我們的 4 類映射
        self.emotion2vec_to_4class = {
            'angry': 'angry',
            'disgusted': 'neutral',
            'fearful': 'neutral',
            'happy': 'happy',
            'neutral': 'neutral',
            'other': 'neutral',
            'sad': 'sad',
            'surprised': 'neutral',
            'unknown': 'neutral'
        }
        
        self.emotion_labels = ["neutral", "happy", "sad", "angry"]
        
        # Initialize models
        self._init_models()
    
    
    def _init_models(self):
        """Initialize emotion2vec and Whisper"""
        print("正在載入 emotion2vec 模型...")

        try:
            from funasr import AutoModel

            self.emotion_model = AutoModel(
                model="iic/emotion2vec_plus_large",  # 改用 plus_large
                hub="ms"
            )
            print("✓ emotion2vec_plus_large 模型載入成功")

        except Exception as e:
            import traceback
            print(f"警告：無法載入 emotion2vec: {e}")
            traceback.print_exc()
            self.emotion_model = None

        # Whisper for forced alignment
        print("載入 Whisper 模型...")
        self.whisper_model = whisper.load_model("large")
        print("✓ Whisper 模型載入成功")
        
    def parse_wdseg(self, wdseg_path):
        """
        Convert HTK wdseg frame alignment → word-level timestamps.
        Each line looks like:
            SFrm  EFrm  Score  Word
            0     2     -117721 <s>
            3    20     -1052907 YOU'RE(2)
        Frame = 10ms = 0.01 seconds
        """
        words = []
        if not Path(wdseg_path).exists():
            return words  # 返回空列表，讓 fallback 處理
        
        with open(wdseg_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                # skip header
                if len(parts) < 4 or parts[0] == "SFrm":
                    continue
                sfrm = int(parts[0])
                efrm = int(parts[1])
                word = parts[3]
                # clean e.g., WORD(2) → WORD
                word = re.sub(r"\(\d+\)$", "", word)
                # remove <sil>, <s>, </s>
                if word in ["<s>", "</s>", "<sil>"]:
                    continue
                start = sfrm * 0.01
                end = (efrm + 1) * 0.01     # HTK frames are inclusive
                words.append((word.lower(), start, end))  # 返回 tuple 格式
        return words
    
    def parse_iemocap_labels(self, session_path):
        """Parse IEMOCAP emotion labels"""
        label_file = session_path / "dialog" / "EmoEvaluation"
        utterance_data = {}
        
        for eval_file in label_file.glob("*.txt"):
            with open(eval_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('['):
                        match = re.match(r'\[([\d.]+)\s*-\s*([\d.]+)\]\s+(\S+)\s+(\w+)', line)
                        if match:
                            start, end, utt_id, emotion = match.groups()
                            mapped_emotion = self.emotion_map.get(emotion, 'neutral')
                            
                            utterance_data[utt_id] = {
                                'start': float(start),
                                'end': float(end),
                                'sentence_emotion': mapped_emotion,
                                'original_emotion': emotion
                            }
        
        return utterance_data
    
    def parse_iemocap_transcripts(self, session_path):
        """Parse IEMOCAP transcripts"""
        transcript_file = session_path / "dialog" / "transcriptions"
        transcripts = {}
        
        for trans_file in transcript_file.glob("*.txt"):
            with open(trans_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    
                    match = re.match(r'(\S+)\s+\[([\d.]+)-([\d.]+)\]:\s*(.*)', line)
                    if match:
                        utt_id, start, end, text = match.groups()
                        transcripts[utt_id] = {
                            'text': text.strip(),
                            'start': float(start),
                            'end': float(end)
                        }
        
        return transcripts
    
    def get_audio_path(self, session_path, utt_id):
        """Get audio file path"""
        dialog_name = '_'.join(utt_id.split('_')[:-1])
        audio_path = session_path / "sentences" / "wav" / dialog_name / f"{utt_id}.wav"
        return audio_path if audio_path.exists() else None
    
    def extract_frame_emotions_emotion2vec(self, audio_path):
        """
        使用 emotion2vec_plus_large 提取 frame-level 情緒
        """
        if self.emotion_model is None:
            return [], []
        
        try:
            # 取得 utterance-level 結果
            result_utt = self.emotion_model.generate(
                str(audio_path),
                output_dir="./temp_outputs",
                granularity="utterance",
                extract_embedding=True
            )
            
            # 取得 frame-level embeddings
            result_frame = self.emotion_model.generate(
                str(audio_path),
                output_dir="./temp_outputs",
                granularity="frame",
                extract_embedding=True
            )
            
            if not result_utt or not result_frame:
                return [], []
            
            scores = np.array(result_utt[0]['scores'])
            labels = result_utt[0]['labels']
            frame_feats = result_frame[0]['feats']
            
            if frame_feats is None or len(frame_feats) == 0:
                return [], []
            
            # 情緒映射
            label_map = {
                '生气/angry': 'angry', '开心/happy': 'happy',
                '难过/sad': 'sad', '中立/neutral': 'neutral',
                '厌恶/disgusted': 'neutral', '恐惧/fearful': 'neutral',
                '其他/other': 'neutral', '吃惊/surprised': 'neutral',
                '<unk>': 'neutral'
            }
            
            # 找主導情緒
            dominant_idx = np.argmax(scores)
            dominant_score = scores[dominant_idx]
            dominant_emotion = label_map.get(labels[dominant_idx], 'neutral')
            
            num_frames = frame_feats.shape[0]
            
            # 計算每個 frame 的情緒強度（用 L2 norm 作為代理）
            frame_norms = np.linalg.norm(frame_feats, axis=1)
            
            # 正規化到 0-1
            if frame_norms.max() > frame_norms.min():
                frame_intensity = (frame_norms - frame_norms.min()) / (frame_norms.max() - frame_norms.min())
            else:
                frame_intensity = np.ones(num_frames) * 0.5
            
            # 用 intensity 閾值決定情緒
            threshold = np.percentile(frame_intensity, 50)  # 中位數
            
            frame_emotions = []
            for intensity in frame_intensity:
                if intensity > threshold and dominant_emotion != 'neutral' and dominant_score > 0.5:
                    frame_emotions.append(dominant_emotion)
                else:
                    frame_emotions.append('neutral')
            
            # Frame 時間戳 (20ms per frame)
            frame_duration = 0.02
            frame_timestamps = (np.arange(num_frames) * frame_duration).tolist()
            
            return frame_emotions, frame_timestamps
            
        except Exception as e:
            print(f"Error with emotion2vec: {e}")
            import traceback
            traceback.print_exc()
            return [], []
            
        
    def perform_alignment(self, audio_path, transcript, utt_id=None):
        """
        Perform forced alignment using IEMOCAP wdseg (優先) or Whisper (fallback)
        """
        
        # 優先使用 IEMOCAP 原始的 .wdseg 檔案
        if utt_id:
            import re
            m = re.match(r"Ses(\d{2})", utt_id)

            if m:
                session_num = int(m.group(1))          # 01 -> 1
                session_name = f"Session{session_num}"
                dialog_name = '_'.join(utt_id.split('_')[:-1])  # Ses01F_script02_1

                wdseg_path = (
                    self.iemocap_path / session_name / "sentences" /
                    "ForcedAlignment" / dialog_name / f"{utt_id}.wdseg"
                )

                word_alignments = self.parse_wdseg(wdseg_path)
                if word_alignments:
                    return word_alignments
            # 如果 m 不存在，什麼都不做，直接往下走 Whisper fallback

        
        # Fallback: 使用 Whisper forced alignment
        try:
            result = self.whisper_model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language='en'
            )
            
            word_alignments = []
            if 'segments' in result:
                for segment in result['segments']:
                    if 'words' in segment:
                        for word_info in segment['words']:
                            word = word_info['word'].strip()
                            start = word_info['start']
                            end = word_info['end']
                            word_alignments.append((word, start, end))
            
            if word_alignments:
                # print(f"✓ 使用 Whisper: {utt_id if utt_id else 'unknown'}")
                return word_alignments
                
        except Exception as e:
            print(f"⚠️  Whisper alignment 失敗: {e}")
        
        # 最後的 fallback: 均分時間
        print(f"⚠️  使用時間均分: {utt_id if utt_id else 'unknown'}")
        words = transcript.split()
        import soundfile as sf
        wav, sr = sf.read(str(audio_path))
        duration = (len(wav) / sr) if sr else 0.0

        word_duration = duration / len(words) if words else 0
        
        word_alignments = []
        for i, word in enumerate(words):
            start = i * word_duration
            end = (i + 1) * word_duration
            word_alignments.append((word, start, end))
        
        return word_alignments
    
    def assign_emotions_to_words(self, frame_emotions, frame_timestamps, word_alignments, fallback_emotion):
        """
        將 frame-level 情緒分配給詞（多數投票）
        """
        word_pseudo_labels = []
        
        for word, start_time, end_time in word_alignments:
            # 找到這個詞時間範圍內的所有 frames
            word_emotions = []
            
            for timestamp, emotion in zip(frame_timestamps, frame_emotions):
                if start_time <= timestamp <= end_time:
                    word_emotions.append(emotion)
            
            # 多數投票
            if word_emotions:
                emotion_counter = Counter(word_emotions)
                word_emotion = emotion_counter.most_common(1)[0][0]
            else:
                # 如果沒有 frame 匹配，使用句級情緒
                word_emotion = fallback_emotion
            
            word_pseudo_labels.append({
                'word': word,
                'emotion': word_emotion,
                'start': start_time,
                'end': end_time
            })
        
        return word_pseudo_labels
    
    def process_session(self, session_num):
        """Process one IEMOCAP session"""
        session_name = f"Session{session_num}"
        session_path = self.iemocap_path / session_name

        if not session_path.exists():
            print(f"Warning: {session_name} not found")
            return []

        print(f"\nProcessing {session_name}...")

        labels = self.parse_iemocap_labels(session_path)
        transcripts = self.parse_iemocap_transcripts(session_path)

        processed_data = []
        VALID_EMOTIONS = {'neu', 'hap', 'exc', 'sad', 'ang'}
        
        # 只有一個迴圈，同時過濾和處理
        for utt_id in tqdm(labels.keys(), desc=f"Session {session_num}"):
            original_emotion = labels[utt_id]['original_emotion']
            
            # 只保留論文使用的情緒
            if original_emotion not in VALID_EMOTIONS:
                continue
                
            if utt_id not in transcripts:
                continue

            audio_path = self.get_audio_path(session_path, utt_id)
            if audio_path is None or not audio_path.exists():
                continue

            
            # ======== 🔧 限制音訊長度至 30 秒以內（不依賴 torchaudio/torchcodec）========
            try:
                import soundfile as sf
                wav, sr = sf.read(str(audio_path))  # wav: (T,) or (T, C)

                max_duration_sec = 30
                max_samples = int(sr * max_duration_sec)

                # 只裁切，不做重取樣；IEMOCAP wav 通常可直接處理
                if len(wav) > max_samples:
                    wav = wav[:max_samples]

                    tmp_path = self.output_path / "trimmed_audio"
                    tmp_path.mkdir(exist_ok=True)
                    trimmed_path = tmp_path / f"{utt_id}.wav"
                    sf.write(str(trimmed_path), wav, sr)
                    audio_path = trimmed_path

            except Exception as e:
                print(f"⚠️ 無法載入或裁切 {audio_path}: {e}")
                continue
            # ========================================================================

            transcript = transcripts[utt_id]['text']
            sentence_emotion = labels[utt_id]['sentence_emotion']

            try:
                # Step 1: Frame-level emotion prediction
                frame_emotions, frame_timestamps = self.extract_frame_emotions_emotion2vec(audio_path)

                # Step 2: Forced alignment
                word_alignments = self.perform_alignment(audio_path, transcript, utt_id=utt_id)

                # Step 3: Assign emotions to words
                if len(frame_emotions) > 0:
                    word_pseudo_labels = self.assign_emotions_to_words(
                        frame_emotions, frame_timestamps, word_alignments, sentence_emotion
                    )
                else:
                    word_pseudo_labels = []
                    for word, start, end in word_alignments:
                        word_pseudo_labels.append({
                            'word': word,
                            'emotion': sentence_emotion,
                            'start': start,
                            'end': end
                        })

                processed_data.append({
                    'utterance_id': utt_id,
                    'audio_path': str(audio_path),
                    'transcript': transcript,
                    'sentence_emotion': sentence_emotion,
                    'word_level_emotions': word_pseudo_labels,
                    'session': session_num
                })

            except Exception as e:
                print(f"Error: {utt_id}: {e}")
                continue

        return processed_data  

    
    def process_session_and_save(self, session_num):
        session_data = self.process_session(session_num)
        out_file = self.output_path / f"Session{session_num}.json"
        with open(out_file, "w") as f:
            json.dump(session_data, f, indent=2)
        print(f"✓ Saved Session {session_num} → {out_file}")
        return session_data
        
    def print_statistics(self, data):
        """Print statistics"""
        print("\n" + "="*60)
        print("Dataset Statistics")
        print("="*60)
        print(f"Total utterances: {len(data)}")
        
        sentence_emotions = [d['sentence_emotion'] for d in data]
        print("\nSentence-level emotion distribution:")
        for emotion in self.emotion_labels:
            count = sentence_emotions.count(emotion)
            print(f"  {emotion:8s}: {count:5d} ({count/len(data)*100:5.1f}%)")
        
        total_words = sum(len(d['word_level_emotions']) for d in data)
        print(f"\nTotal words: {total_words}")
        
        word_emotions = []
        for d in data:
            word_emotions.extend([w['emotion'] for w in d['word_level_emotions']])
        
        print("\nWord-level emotion distribution:")
        for emotion in self.emotion_labels:
            count = word_emotions.count(emotion)
            print(f"  {emotion:8s}: {count:6d} ({count/len(word_emotions)*100:5.1f}%)")
        
        print("="*60)

def main():
    IEMOCAP_PATH = "/home/ouo/whisper_emotion/workspace/iemocap/IEMOCAP_full_release"
    OUTPUT_PATH = "/home/ouo/whisper_emotion/workspace/iemocap_processed"
    
    preprocessor = IEMOCAPPreprocessor(
        iemocap_path=IEMOCAP_PATH,
        output_path=OUTPUT_PATH
    )
    
    # 產全部（train set）
    for session in [2,3,4,5]:
        preprocessor.process_session_and_save(session)

    print("\n✓ 全部 Session 預處理完成")


if __name__ == "__main__":
    main()


