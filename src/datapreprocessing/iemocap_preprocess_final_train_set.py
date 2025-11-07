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
            
            # 使用 emotion2vec+ large 模型
            self.emotion_model = AutoModel(
                model="iic/emotion2vec_base",
                hub="ms"  # 使用 modelscope
            )
            
            print("✓ emotion2vec 模型載入成功")
            
        except Exception as e:
            print(f"警告：無法載入 emotion2vec: {e}")
            print("將使用原始標註作為後備方案")
            self.emotion_model = None
        
        # Whisper for forced alignment
        print("載入 Whisper 模型...")
        self.whisper_model = whisper.load_model("large")
        print("✓ Whisper 模型載入成功")
        
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
        使用 emotion2vec 提取 frame-level 情緒
        """
        if self.emotion_model is None:
            return [], []
        
        try:
            # 使用 emotion2vec 進行 frame-level 預測
            result = self.emotion_model.generate(
                str(audio_path),
                output_dir="./temp_outputs",
                granularity="frame",
                extract_embedding=False
            )
            
            # emotion2vec 中英文標籤映射
            chinese_to_english = {
                '生气': 'angry',
                '厌恶': 'disgusted',
                '恐惧': 'fearful',
                '开心': 'happy',
                '中立': 'neutral',
                '其他': 'other',
                '难过': 'sad',
                '吃惊': 'surprised',
                '<unk>': 'unknown'
            }
            
            # 解析結果
            if result and len(result) > 0:
                first_result = result[0]
                
                # emotion2vec 返回的是 list of Chinese labels
                if isinstance(first_result, list):
                    frame_labels = first_result
                elif isinstance(first_result, dict) and 'labels' in first_result:
                    frame_labels = first_result['labels']
                else:
                    print(f"Unexpected result format: {type(first_result)}")
                    return [], []
                
                # 轉換中文標籤為 4 類情緒
                frame_emotions = []
                for chinese_label in frame_labels:
                    # 轉換中文到英文
                    english_label = chinese_to_english.get(chinese_label, 'unknown')
                    # 映射到 4 類
                    mapped_emotion = self.emotion2vec_to_4class.get(english_label, 'neutral')
                    frame_emotions.append(mapped_emotion)
                
                # 計算 frame 時間戳（假設 50Hz）
                frame_duration = 0.02  # 20ms
                frame_timestamps = np.arange(len(frame_emotions)) * frame_duration
                
                return frame_emotions, frame_timestamps
            
            return [], []
            
        except Exception as e:
            print(f"Error with emotion2vec: {e}")
            import traceback
            traceback.print_exc()
            return [], []
            
        
    def perform_alignment(self, audio_path, transcript):
        """Perform forced alignment using Whisper"""
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
            
            return word_alignments
            
        except Exception as e:
            # Fallback: 均分時間
            words = transcript.split()
            waveform, sr = torchaudio.load(audio_path)
            duration = waveform.shape[1] / sr
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
        
        for utt_id in tqdm(labels.keys(), desc=f"Session {session_num}"):
            if utt_id not in transcripts:
                continue
            
            audio_path = self.get_audio_path(session_path, utt_id)
            if audio_path is None or not audio_path.exists():
                continue
            
            transcript = transcripts[utt_id]['text']
            sentence_emotion = labels[utt_id]['sentence_emotion']
            
            try:
                # Step 1: Frame-level emotion prediction with emotion2vec
                frame_emotions, frame_timestamps = self.extract_frame_emotions_emotion2vec(audio_path)
                
                # Step 2: Forced alignment
                word_alignments = self.perform_alignment(audio_path, transcript)
                
                # Step 3: Assign emotions to words
                if len(frame_emotions) > 0:
                    # 使用 emotion2vec 的 frame-level 預測
                    word_pseudo_labels = self.assign_emotions_to_words(
                        frame_emotions, frame_timestamps, word_alignments, sentence_emotion
                    )
                else:
                    # 後備方案：使用句級情緒
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
    
    def process_train_sessions(self, sessions=[ 2, 3, 4]):
        """Process part of  sessions"""
        all_data = []
        
        for session_num in sessions:
            session_data = self.process_session(session_num)
            all_data.extend(session_data)
        
        output_file = self.output_path / "iemocap_with_pseudo_labels_train_set.json"
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"\n✓ Processed {len(all_data)} utterances")
        print(f"✓ Saved to {output_file}")
        
        self.print_statistics(all_data)
        
        return all_data
    
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
    
    processed_data = preprocessor.process_train_sessions(sessions=[ 2, 3, 4])
    
    print("\n✓ Preprocessing complete!")

if __name__ == "__main__":
    main()

