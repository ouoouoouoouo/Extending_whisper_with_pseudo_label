import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, Audio
import librosa
import numpy as np
from tqdm.auto import tqdm
import math # 用於計算時長

# --- (與 generate_multitask_targets.py 相同的類別) ---

class WhisperAudioProcessor:
    """處理音訊：載入、截斷、填充"""
    
    def __init__(self, feature_extractor: WhisperFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.sampling_rate = 16000
        self.max_audio_length_seconds = 30
        self.max_audio_samples = self.max_audio_length_seconds * self.sampling_rate
        self.max_feature_length = 3000 
        print(f"✓ (CV) 音訊處理器：音訊將被截斷/填充至 {self.max_audio_length_seconds} 秒")

    def process_audio(self, audio_path: str) -> Dict[str, np.ndarray]:
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            if len(audio) > self.max_audio_samples:
                audio = audio[:self.max_audio_samples]
                
            input_features = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="np"
            ).input_features
            
            input_features = input_features.squeeze(0) 
            feature_length = input_features.shape[1]
            
            if feature_length > self.max_feature_length:
                 input_features = input_features[:, :self.max_feature_length]
            elif feature_length < self.max_feature_length:
                padding = np.zeros((input_features.shape[0], self.max_feature_length - feature_length), dtype=input_features.dtype)
                input_features = np.concatenate([input_features, padding], axis=1)

            return {"input_features": input_features}
        except Exception as e:
            print(f"⚠ 處理音訊 {audio_path} 失敗: {e}")
            raise e # 讓上層捕捉

class CommonVoicePreprocessor:
    """
    用於處理 Common Voice 資料集，轉換為 *標準* Whisper ASR 格式
    
    ** (V2: 支援從 validated.tsv 自動切分) **
    """
    
    def __init__(
        self,
        base_model: str = "openai/whisper-large-v2",
        cv_data_path: str = "/home/ouo/whisper_emotion/workspace/commonvoice/cv-corpus-21.0-delta-2025-03-14/en"
    ):
        print("="*80)
        print("初始化 Common Voice 預處理器 (V2 - 自動切分)")
        print("="*80)
        
        self.base_path = Path(cv_data_path)
        self.clips_path = self.base_path / "clips"
        
        # 1. 載入標準 Tokenizer 和 Feature Extractor
        self.tokenizer = WhisperTokenizer.from_pretrained(base_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
        
        # 2. 準備音訊處理器
        self.audio_processor = WhisperAudioProcessor(self.feature_extractor)
        
        # 3. 準備標準 ASR 提示
        try:
            self.prefix_ids = [
                self.tokenizer.bos_token_id, # <|startoftranscript|>
                self.tokenizer.convert_tokens_to_ids("<|en|>"),
                self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
                self.tokenizer.convert_tokens_to_ids("<|notimestamps|>") # 推薦加入以簡化 ASR
            ]
            self.prefix_labels = self.prefix_ids[1:] # Labels 不包含 SOT
            
            print(f"✓ 標準 ASR 提示 (SOT, en, transcribe, notimestamps) 已設定")
        except KeyError as e:
            print(f"錯誤：Tokenizer 中缺少標準 Whisper token: {e}")
            raise e

    def load_tsv_data(self, tsv_name: str) -> pd.DataFrame:
        tsv_path = self.base_path / tsv_name
        print(f"正在讀取 TSV: {tsv_path}")
        try:
            # 嘗試讀取 'path' 和 'sentence'
            df = pd.read_csv(tsv_path, sep='\t', quoting=3, usecols=['path', 'sentence'])
        except ValueError:
            # 如果 'sentence' 不存在，嘗試 'transcript'
            print("警告: 找不到 'sentence' 欄位，嘗試 'transcript' 欄位...")
            try:
                df = pd.read_csv(tsv_path, sep='\t', quoting=3, usecols=['path', 'transcript'])
                df = df.rename(columns={'transcript': 'sentence'}) # 重新命名以便後續使用
            except ValueError:
                print(f"錯誤: {tsv_path} 中必須包含 'path' 欄位以及 'sentence' 或 'transcript' 欄位")
                raise
        
        df = df.dropna() # 移除遺失值
        print(f"✓ 從 {tsv_name} 載入 {len(df)} 筆有效樣本")
        return df

    def create_target_sequence(self, transcript: str) -> List[int]:
        """建立 *標準* ASR 目標序列"""
        
        custom_ids = self.tokenizer.encode(
            transcript,
            add_special_tokens=False 
        )
        
        # 格式: [EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]
        target_ids = self.prefix_labels + custom_ids + [self.tokenizer.eos_token_id]
        return target_ids
    
    def create_decoder_input_ids(self, target_ids: List[int]) -> List[int]:
        """建立 decoder_input_ids (SOT + 標籤但不含 EOS)"""
        # target_ids = [EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]
        # decoder_input_ids = [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ...]
        
        # 我們需要 SOT + target_ids (不含 EOS)
        # SOT = self.prefix_ids[0]
        # target_ids_no_eos = target_ids[:-1]
        # return [self.prefix_ids[0]] + target_ids_no_eos
        
        # --- 修正：更簡單的 Whisper 標準格式 ---
        # labels = [SOT, EN, TRANSCRIBE, ..., EOS]
        # decoder_input_ids = [SOT, EN, TRANSCRIBE, ...] (不含 EOS)
        
        # 讓我們使用 `preprocess_common_voice.py` V1 的邏輯，它更標準
        # 這裡的 self.prefix_ids 包含 SOT
        
        custom_ids = self.tokenizer.encode(transcript, add_special_tokens=False)
        
        # 標籤 (Labels)
        # [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]
        labels = self.prefix_ids + custom_ids + [self.tokenizer.eos_token_id]
        
        # 解碼器輸入 (Decoder Input IDs)
        # [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ...] (不含 EOS)
        decoder_input_ids = labels[:-1]
        
        return labels, decoder_input_ids

    def preprocess_single_example(self, audio_file_path: str, transcript: str) -> Dict:
        """預處理單一 Common Voice 樣本"""
        
        # 1. 處理音訊
        audio_features = self.audio_processor.process_audio(audio_file_path)
        
        # 2. 建立目標序列
        custom_ids = self.tokenizer.encode(transcript, add_special_tokens=False)
        
        # 3. 建立完整序列
        # [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]
        full_sequence = self.prefix_ids + custom_ids + [self.tokenizer.eos_token_id]
        
        # 4. 建立 decoder_input_ids（去掉最後的 EOS）
        decoder_input_ids = full_sequence[:-1]
        # [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ...]
    
        
        # 5. 建立 labels（去掉第一個 SOT，保留 EOS）- 已 shift
        labels = full_sequence[1:]
        # [EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]

        
        
        
        # 6. (可選) 將前綴 tokens 設為 -100，不計算 loss
        labels_with_prefix_ignored = labels.copy()
        labels_with_prefix_ignored[0] = -100  # <|en|>
        labels_with_prefix_ignored[1] = -100  # <|transcribe|>
        labels_with_prefix_ignored[2] = -100  # <|notimestamps|>
    
        
        return {
            'input_features': audio_features['input_features'],
            'labels': labels_with_prefix_ignored,
            'decoder_input_ids': decoder_input_ids,
            'original_text': transcript
        }
    
    def process_and_save(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        split_name: str
    ):
        """
        處理 DataFrame 並儲存為 Hugging Face Dataset
        """
        processed_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"處理 {split_name}"):
            try:
                audio_file = self.clips_path / row['path']
                transcript = row['sentence']
                
                processed = self.preprocess_single_example(str(audio_file), transcript)
                processed_data.append(processed)
            except Exception as e:
                # 忽略處理失敗的音訊
                # print(f"警告：跳過 {row['path']}，原因: {e}")
                continue
        
        dataset = Dataset.from_list(processed_data)
        
        print(f"\n✓ 成功處理 {len(dataset)} 筆 {split_name} 資料")
        
        dataset.save_to_disk(output_path)
        print(f"✓ {split_name} 資料集已儲存至: {output_path}")

    # --- ↓↓↓ 這是新增的、遺失的方法 ↓↓↓ ---
    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        建立 DataLoader
        """
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            """
            Common Voice 的自定義 collate function
            (只包含 'input_features', 'labels', 'decoder_input_ids', 'original_text')
            """
            # 取得 batch 中最長的序列長度
            max_label_length = max(len(item['labels']) for item in batch)
            max_decoder_length = max(len(item['decoder_input_ids']) for item in batch)
            
            # input_features 已被填充到 (80, 3000)，可以直接堆疊
            input_features = torch.stack([
                torch.tensor(item['input_features']) for item in batch
            ])
            
            # Padding labels 和 decoder_input_ids
            labels = []
            decoder_input_ids = []
            
            for item in batch:
                # Padding labels
                label = torch.tensor(item['labels'], dtype=torch.long)
                len_label = len(label)
                padded_label = torch.cat([
                    label,
                    torch.full((max_label_length - len_label,), -100, dtype=torch.long)
                ])
                labels.append(padded_label)
                
                # Padding decoder_input_ids
                decoder_ids = torch.tensor(item['decoder_input_ids'], dtype=torch.long)
                len_decoder_ids = len(decoder_ids)
                padded_decoder = torch.cat([
                    decoder_ids,
                    torch.full(
                        (max_decoder_length - len_decoder_ids,),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long
                    )
                ])
                decoder_input_ids.append(padded_decoder)
            
            return {
                'input_features': input_features,
                'labels': torch.stack(labels),
                'decoder_input_ids': torch.stack(decoder_input_ids),
                'original_texts': [item['original_text'] for item in batch]
                # 注意：CV 資料沒有 'target_text' 或 'sentence_emotions'
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    # --- ↑↑↑ 新增的方法結束 ↑↑↑ ---

def main():
    CV_PATH = "/home/ouo/whisper_emotion/workspace/commonvoice/cv-corpus-21.0-delta-2025-03-14/en"
    
    preprocessor = CommonVoicePreprocessor(
        base_model="openai/whisper-large-v2",
        cv_data_path=CV_PATH
    )
    
    # 載入數據 - 改用 other.tsv
    try:
        df_all = preprocessor.load_tsv_data("other.tsv")  # ← 改這裡！
    except FileNotFoundError:
        print(f"錯誤: 找不到 'other.tsv'")
        return
    
    # 隨機打亂
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    total_samples = len(df_all)
    print(f"\n總樣本數: {total_samples:,}")
    
    # 切分（論文要求：10h 訓練 + 1h 驗證 + 1h 測試）
    AVERAGE_DURATION_SECONDS = 5
    samples_per_hour = int(3600 / AVERAGE_DURATION_SECONDS)
    
    test_size = samples_per_hour    # 1 小時 = ~720 樣本
    val_size = samples_per_hour     # 1 小時 = ~720 樣本  
    train_size = samples_per_hour * 10  # 10 小時 = ~7200 樣本
    
    # 確保不超過總數
    if test_size + val_size + train_size > total_samples:
        print(f"警告: 切分大小 ({test_size + val_size + train_size}) 超過總數 ({total_samples})")
        # 按比例調整
        train_size = int(total_samples * 0.833)  # 83.3%
        val_size = int(total_samples * 0.083)    # 8.3%
        test_size = total_samples - train_size - val_size
    
    print(f"\n將切分為:")
    print(f"  訓練: {train_size:,} 樣本 (~{train_size * AVERAGE_DURATION_SECONDS / 3600:.1f} 小時)")
    print(f"  驗證: {val_size:,} 樣本 (~{val_size * AVERAGE_DURATION_SECONDS / 3600:.1f} 小時)")
    print(f"  測試: {test_size:,} 樣本 (~{test_size * AVERAGE_DURATION_SECONDS / 3600:.1f} 小時)")
    
    # 執行切分
    test_df = df_all.iloc[:test_size]
    val_df = df_all.iloc[test_size:test_size + val_size]
    train_df = df_all.iloc[test_size + val_size:test_size + val_size + train_size]
    
    print(f"\n實際切分大小:")
    print(f"  訓練: {len(train_df):,}")
    print(f"  驗證: {len(val_df):,}")
    print(f"  測試: {len(test_df):,}")
    
    # 處理並儲存
    print(f"\n開始處理...")
    preprocessor.process_and_save(train_df, "./cv_processed/train", "CV Train")
    preprocessor.process_and_save(val_df, "./cv_processed/val", "CV Val")
    preprocessor.process_and_save(test_df, "./cv_processed/test", "CV Test")
    
    print("\n✓ Common Voice 預處理完成！")

if __name__ == "__main__":
    main()