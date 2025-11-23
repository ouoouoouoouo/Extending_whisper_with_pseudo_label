# preprocess_common_voice_for_rehearsal.py
"""
Common Voice 預處理 - 用於 Rehearsal Learning
**重要**: 必須在 IEMOCAP 預處理之後執行，使用相同的自定義 tokenizer
"""

import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, concatenate_datasets
import librosa
import numpy as np
from tqdm.auto import tqdm
from pydub import AudioSegment


import librosa
import soundfile as sf



class WhisperAudioProcessor:
    """處理音訊：載入、截斷、填充（使用 soundfile + librosa）"""

    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.sampling_rate = 16000

        # 改為 30 秒上限（Whisper 標準）
        self.max_audio_length_seconds = 30
        self.max_audio_samples = self.max_audio_length_seconds * self.sampling_rate

        # Whisper 30 秒 → 3000 frames
        self.max_feature_length = 3000

        print(f"✓ 音訊處理器：上限 {self.max_audio_length_seconds}s (≈{self.max_feature_length} frames)")

    def process_audio(self, audio_path: str) -> Dict[str, np.ndarray]:
        try:
            # 優先使用 librosa
            try:
                audio, sr = librosa.load(audio_path, sr=self.sampling_rate, mono=True)
            except Exception:
                # 若失敗，用 pydub 重新載入 MP3
                audio_segment = AudioSegment.from_file(audio_path)
                audio_segment = audio_segment.set_frame_rate(self.sampling_rate).set_channels(1)
                audio = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                sr = self.sampling_rate

            # ===== 關鍵修改：截斷到 15 秒 =====
            if len(audio) > self.max_audio_samples:
                audio = audio[:self.max_audio_samples]

            # ===== 轉成 Whisper 特徵 =====
            input_features = self.feature_extractor(
                audio,
                sampling_rate=self.sampling_rate,
                return_tensors="np"
            ).input_features

            input_features = input_features.squeeze(0)
            feature_length = input_features.shape[1]

            # ===== 改為固定補齊到 Whisper 需求的 3000 frame =====
            TARGET_FEATURE_LEN = 3000
            if feature_length > TARGET_FEATURE_LEN:
                input_features = input_features[:, :TARGET_FEATURE_LEN]
            elif feature_length < TARGET_FEATURE_LEN:
                padding = np.zeros(
                    (input_features.shape[0], TARGET_FEATURE_LEN - feature_length),
                    dtype=input_features.dtype
                )
                input_features = np.concatenate([input_features, padding], axis=1)
            return {"input_features": input_features}

        except Exception as e:
            print(f"⚠ 處理音訊 {audio_path} 失敗: {e}")
            raise e

        

class CommonVoicePreprocessorForRehearsal:
    """
    Common Voice 預處理器 - 用於 Rehearsal Learning
    使用與 IEMOCAP 相同的自定義 tokenizer，但只輸出標準 ASR 格式
    """
    
    def __init__(
        self,
        custom_tokenizer_path: str,  # ← 使用自定義 tokenizer
        base_model: str = "openai/whisper-large-v2",
        cv_data_path: str = None
    ):
        print("="*80)
        print("初始化 Common Voice 預處理器 (用於 Rehearsal Learning)")
        print("="*80)
        
        self.base_path = Path(cv_data_path)
        self.clips_path = self.base_path / "clips"
        
        # ===== 關鍵：載入自定義 Tokenizer =====
        print(f"載入自定義 Tokenizer: {custom_tokenizer_path}")
        self.tokenizer = WhisperTokenizer.from_pretrained(custom_tokenizer_path)
        print(f"✓ Tokenizer 詞彙表大小: {len(self.tokenizer)}")
        
        # 檢查是否包含情緒 tokens（確保使用了正確的 tokenizer）
        test_tokens = ['<|sle_happy|>', '<|wle_happy|>']
        for token in test_tokens:
            if token in self.tokenizer.get_vocab():
                print(f"  ✓ 確認包含情緒 token: {token}")
            else:
                print(f"  ⚠ 警告：缺少情緒 token: {token}")
        
        # 載入 Feature Extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
        self.audio_processor = WhisperAudioProcessor(self.feature_extractor)
        
        # ===== 定義標準 ASR 前綴（Format F）=====
        try:
            self.prefix_ids = [
                self.tokenizer.bos_token_id,  # <|startoftranscript|>
                self.tokenizer.convert_tokens_to_ids("<|en|>"),
                self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
                self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
            ]
            print(f"✓ 標準 ASR 前綴: {self.prefix_ids}")
        except KeyError as e:
            print(f"錯誤：Tokenizer 中缺少標準 Whisper token: {e}")
            raise e

    def load_tsv_data(self, tsv_name: str) -> pd.DataFrame:
        """載入 Common Voice TSV"""
        tsv_path = self.base_path / tsv_name
        print(f"正在讀取 TSV: {tsv_path}")
        
        try:
            df = pd.read_csv(tsv_path, sep='\t', quoting=3, usecols=['path', 'sentence'])
        except ValueError:
            try:
                df = pd.read_csv(tsv_path, sep='\t', quoting=3, usecols=['path', 'transcript'])
                df = df.rename(columns={'transcript': 'sentence'})
            except ValueError:
                print(f"錯誤: TSV 必須包含 'path' 和 'sentence'/'transcript' 欄位")
                raise
        
        df = df.dropna()
        print(f"✓ 從 {tsv_name} 載入 {len(df)} 筆有效樣本")
        return df

    def preprocess_single_example(self, audio_file_path: str, transcript: str) -> Dict:
        """
        預處理單一 Common Voice 樣本
        格式 F: [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]
        """
        
        # 1. 處理音訊
        audio_features = self.audio_processor.process_audio(audio_file_path)
        
        # 2. Tokenize 文本
        custom_ids = self.tokenizer.encode(transcript, add_special_tokens=False)
        
        # 3. 建立完整序列 (Format F)
        # [SOT, EN, TRANSCRIBE, NOTIMESTAMPS, WORD1, ..., EOS]
        full_sequence = self.prefix_ids + custom_ids + [self.tokenizer.eos_token_id]
        
        # 4. Decoder input (去掉最後的 EOS)
        decoder_input_ids = full_sequence[:-1]
        
        # 5. Labels (去掉第一個 SOT，保留 EOS)
        labels = full_sequence[1:]
        
        # 6. 將前綴 tokens 設為 -100（不計算 loss）
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
    
    from datasets import Dataset, concatenate_datasets

    def process_and_save(self, df: pd.DataFrame, output_path: str, split_name: str):
        """處理 DataFrame 並分批儲存 Dataset（避免 2GB 限制）"""
        processed_data = []
        datasets_list = []
        batch_size = 2000  # 每次處理 2000 筆（可視記憶體大小調整）

        print(f"\n開始處理 {split_name}，共 {len(df)} 筆樣本...")
        for i, row in tqdm(df.iterrows(), total=len(df), desc=f"處理 {split_name}"):
            try:
                audio_file = self.clips_path / row['path']
                transcript = row['sentence']
                processed = self.preprocess_single_example(str(audio_file), transcript)
                
                # 🔹 節省記憶體：壓縮 input_features 為 float16
                processed["input_features"] = processed["input_features"].astype(np.float16)
                processed_data.append(processed)
            except Exception as e:
                print(f"⚠️ 無法處理 {row['path']}: {e}")
                continue

            # 🔹 每 batch_size 筆資料就 flush 一次
            if len(processed_data) >= batch_size:
                try:
                    sub_dataset = Dataset.from_list(processed_data)
                    datasets_list.append(sub_dataset)
                    processed_data = []  # 清空暫存
                except Exception as e:
                    print(f"⚠️ 建立子資料集失敗: {e}")

        # 🔹 處理最後殘餘資料
        if len(processed_data) > 0:
            sub_dataset = Dataset.from_list(processed_data)
            datasets_list.append(sub_dataset)

        if len(datasets_list) == 0:
            print(f"⚠ 無法儲存 {split_name}: 無有效樣本")
            return None

        # 🔹 合併所有子 Dataset
        dataset = concatenate_datasets(datasets_list)
        dataset.save_to_disk(output_path)

        print(f"✓ {split_name} 資料集已儲存至: {output_path} ({len(dataset)} 筆樣本)")
        return dataset


    def create_dataloader(self, dataset: Dataset, batch_size: int = 4, shuffle: bool = True, num_workers: int = 4):
        """建立 DataLoader"""
        from torch.utils.data import DataLoader
        
        def collate_fn(batch):
            max_label_length = max(len(item['labels']) for item in batch)
            max_decoder_length = max(len(item['decoder_input_ids']) for item in batch)
            
            input_features = torch.stack([
                torch.tensor(item['input_features']) for item in batch
            ])
            
            labels = []
            decoder_input_ids = []
            
            for item in batch:
                # Padding labels
                label = torch.tensor(item['labels'], dtype=torch.long)
                padded_label = torch.cat([
                    label,
                    torch.full((max_label_length - len(label),), -100, dtype=torch.long)
                ])
                labels.append(padded_label)
                
                # Padding decoder_input_ids
                decoder_ids = torch.tensor(item['decoder_input_ids'], dtype=torch.long)
                padded_decoder = torch.cat([
                    decoder_ids,
                    torch.full(
                        (max_decoder_length - len(decoder_ids),),
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
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


def main():
    """
    主執行流程
    **重要**: 必須先執行 IEMOCAP 預處理（generate_multitask_targets.py）
    """
    
    # ===== 配置 =====
    CUSTOM_TOKENIZER_PATH = "./custom_whisper_tokenizer"  # ← 使用 IEMOCAP 建立的 tokenizer
    CV_DATA_PATH = "/home/ouo/whisper_emotion/workspace/CV/en"
    OUTPUT_BASE = "./cv_processed_for_rehearsal"
    
    # 檢查 tokenizer 是否存在
    if not Path(CUSTOM_TOKENIZER_PATH).exists():
        print(f"錯誤: 找不到自定義 tokenizer: {CUSTOM_TOKENIZER_PATH}")
        print("請先執行 IEMOCAP 預處理腳本 (generate_multitask_targets.py)")
        return
    
    # ===== 初始化預處理器 =====
    preprocessor = CommonVoicePreprocessorForRehearsal(
        custom_tokenizer_path=CUSTOM_TOKENIZER_PATH,
        base_model="openai/whisper-large-v2",
        cv_data_path=CV_DATA_PATH
    )
    
    # ===== 載入並切分資料 =====
    try:
        df_all = preprocessor.load_tsv_data("validated.tsv")
    except FileNotFoundError:
        print("錯誤: 找不到 'validated.tsv'")
        return
    
    # 隨機打亂
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 計算切分大小（論文：10h train + 1h dev + 1h test）
    AVERAGE_DURATION_SECONDS = 2
    samples_per_hour = int(3600 / AVERAGE_DURATION_SECONDS)
    
    train_size = samples_per_hour * 10  # 10 小時
    val_size = samples_per_hour         # 1 小時
    test_size = samples_per_hour        # 1 小時
    
    total_needed = train_size + val_size + test_size
    if total_needed > len(df_all):
        print(f"警告: 需要 {total_needed} 樣本，但只有 {len(df_all)} 樣本")
        # 按比例調整
        train_size = int(len(df_all) * 0.833)
        val_size = int(len(df_all) * 0.083)
        test_size = len(df_all) - train_size - val_size
    
    print(f"\n切分大小:")
    print(f"  訓練: {train_size} 樣本")
    print(f"  驗證: {val_size} 樣本")
    print(f"  測試: {test_size} 樣本")
    
    # 執行切分
    test_df = df_all.iloc[:test_size]
    val_df = df_all.iloc[test_size:test_size + val_size]
    train_df = df_all.iloc[test_size + val_size:test_size + val_size + train_size]
    
    # ===== 處理並儲存 =====
    print("\n開始處理...")
    preprocessor.process_and_save(train_df, f"{OUTPUT_BASE}/train", "CV Train")
    preprocessor.process_and_save(val_df, f"{OUTPUT_BASE}/val", "CV Val")
    preprocessor.process_and_save(test_df, f"{OUTPUT_BASE}/test", "CV Test")
    
    print("\n" + "="*80)
    print("✓ Common Voice 預處理完成（用於 Rehearsal Learning）")
    print("="*80)
    print(f"輸出目錄: {OUTPUT_BASE}")
    print(f"Tokenizer: {CUSTOM_TOKENIZER_PATH}")
    print("\n現在可以進行聯合訓練了！")


if __name__ == "__main__":
    main()

