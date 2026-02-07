import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Union # <-- 新增 Union
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, Audio, concatenate_datasets # <-- 新增 concatenate_datasets
import librosa
import numpy as np
from tqdm.auto import tqdm # <-- 確保在頂層導入 tqdm


class WhisperEmotionDataPreprocessor:
    """
    用於處理包含 pseudo label 的資料並轉換為 Whisper 模型輸入格式
    """
    
    def __init__(
        self,
        base_model: str = "openai/whisper-large-v2",
        save_tokenizer_path: str = "./custom_whisper_tokenizer",
    ):
        """
        初始化預處理器
        
        Args:
            base_model: 基礎 Whisper 模型名稱
            save_tokenizer_path: 儲存客製化 tokenizer 的路徑
        """
        # 1. 載入基礎 tokenizer
        self.tokenizer = WhisperTokenizer.from_pretrained(base_model)
        
        # 2. 定義情緒特殊 tokens
        self.sle_tokens = {
            'happy': '<|sle_happy|>',
            'sad': '<|sle_sad|>',
            'angry': '<|sle_angry|>',
            'neutral': '<|sle_neutral|>'
        }
        
        self.wle_tokens = {
            'happy': '<|wle_happy|>',
            'sad': '<|wle_sad|>',
            'angry': '<|wle_angry|>',
            'neutral': '<|wle_neutral|>'
        }
        
        # 3. 添加特殊 tokens 到 tokenizer
        special_tokens = list(self.sle_tokens.values()) + list(self.wle_tokens.values())
        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        print(f"✓ 已添加 {num_added} 個特殊 tokens")
        
        # 4. 儲存客製化 tokenizer
        self.save_tokenizer_path = Path(save_tokenizer_path)
        self.save_tokenizer_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(self.save_tokenizer_path))
        print(f"✓ Tokenizer 已儲存至: {self.save_tokenizer_path}")
        
        # 5. 載入 feature extractor (處理音訊)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
        
        # 6. 建立 token ID 映射
        self.sle_token_ids = {
            emotion: self.tokenizer.convert_tokens_to_ids(token)
            for emotion, token in self.sle_tokens.items()
        }
        
        self.wle_token_ids = {
            emotion: self.tokenizer.convert_tokens_to_ids(token)
            for emotion, token in self.wle_tokens.items()
        }
        
        print(f"✓ SLE Token IDs: {self.sle_token_ids}")
        print(f"✓ WLE Token IDs: {self.wle_token_ids}")
        
        # --- 定義最大音訊長度 ---
        self.max_audio_length_seconds = 30
        self.sampling_rate = 16000
        self.max_audio_samples = self.max_audio_length_seconds * self.sampling_rate
        print(f"✓ 音訊將被截斷至 {self.max_audio_length_seconds} 秒 ({self.max_audio_samples} 個取樣點)")

    
    def load_pseudo_label_json(self, json_path: Union[str, List[str]]) -> List[Dict]:
        """
        載入包含 pseudo label 的 JSON 檔案。支援單一檔案或檔案列表。
        """
        # 如果是單一字串，轉換為列表以便統一處理
        if isinstance(json_path, str):
            json_paths = [json_path]
        else:
            json_paths = json_path

        all_data = []
        for path in json_paths:
            print(f"正在載入: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_data.extend(data)
                print(f"  ✓ 成功載入 {len(data)} 筆資料")
            except FileNotFoundError:
                 print(f"  ⚠ 警告: 找不到檔案 {path}，跳過。")

        print(f"✓ 總共載入 {len(all_data)} 筆資料")
        return all_data
    
    def create_target_sequence(
        self,
        transcript: str,
        sentence_emotion: str,
        word_level_emotions: List[Dict[str, str]]
    ) -> Tuple[str, List[int]]:
        """
        建立目標序列，包含標準 Whisper 前綴 和 自訂情緒/文字序列
        格式: <|startoftranscript|> <|en|> <|transcribe|> <|sle_emotion|> word1 <|wle_emotion1|> ... <|endoftext|>
        """
        # --- 1. 定義標準前綴 IDs ---
        try:
            prefix_ids = [
                self.tokenizer.bos_token_id, # <|startoftranscript|>
                self.tokenizer.convert_tokens_to_ids("<|en|>"),
                self.tokenizer.convert_tokens_to_ids("<|transcribe|>")
            ]
        except KeyError as e:
            print(f"錯誤：Tokenizer 中缺少標準 Whisper token: {e}")
            raise e
        
        # --- 2. 構建自訂部分的 tokens (SLE + words + WLE) ---
        custom_tokens = [self.sle_tokens[sentence_emotion]]
        for word_info in word_level_emotions:
            word = word_info['word']
            emotion = word_info['emotion']
            custom_tokens.append(word)
            if emotion != 'neutral':
                custom_tokens.append(self.wle_tokens[emotion])
        
        # --- 3. Tokenize 自訂部分 ---
        target_text = ' '.join(custom_tokens) 
        custom_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False 
        )
        
        # --- 4. 組合最終的 target_ids ---
        target_ids = prefix_ids + custom_ids + [self.tokenizer.eos_token_id]
        
        return target_text, target_ids
    
    def process_audio(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        處理音訊檔案，轉換為 Whisper 所需的特徵 (包含截斷和填充)
        """
        # 載入音訊 (Whisper 使用 16kHz 採樣率)
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        
        # 截斷過長的音訊
        if len(audio) > self.max_audio_samples:
            audio = audio[:self.max_audio_samples]
            
        # 使用 feature extractor 轉換為 log-mel spectrogram
        input_features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="np" # 使用 numpy array
        ).input_features
        
        # input_features shape (1, 80, 3000) -> (80, 3000)
        input_features = input_features.squeeze(0) 

        # 填充過短的音訊
        feature_length = input_features.shape[1]
        if feature_length < 3000:
            padding = np.zeros((input_features.shape[0], 3000 - feature_length), dtype=input_features.dtype)
            input_features = np.concatenate([input_features, padding], axis=1)

        return {
            "input_features": input_features 
        }
    
    def preprocess_single_example(self, example: Dict) -> Dict:
        """
        預處理單一樣本
        """
        # 1. 處理音訊
        audio_features = self.process_audio(example['audio_path'])
        
        # 2. 建立目標序列
        target_text, target_ids = self.create_target_sequence(
            transcript=example['transcript'],
            sentence_emotion=example['sentence_emotion'],
            word_level_emotions=example['word_level_emotions']
        )
        
        # 3. 建立 decoder_input_ids（去掉最後的 EOS）
        decoder_input_ids = target_ids[:-1]
        
        # 4. 建立 labels（去掉第一個 BOS，保留 EOS）
        labels = target_ids[1:]
        
        # 5. 將前綴 tokens 設為 -100，不計算 loss
        labels_with_prefix_ignored = labels.copy()
        labels_with_prefix_ignored[0] = -100  # <lang>
        labels_with_prefix_ignored[1] = -100  # <task>
        
        return {
            'input_features': audio_features['input_features'],
            'labels': labels_with_prefix_ignored,
            'decoder_input_ids': decoder_input_ids,
            'target_text': target_text,
            'original_text': example['transcript'],
            'sentence_emotion': example['sentence_emotion'],
            'audio_path': example['audio_path']
        }
    
    def preprocess_dataset(
        self,
        json_path: Union[str, List[str]], # <-- 修改型別提示
        output_path: str = None
    ) -> Dataset:
        """
        預處理整個資料集。支援單一 JSON 檔案或檔案列表。
        """
        # 1. 載入 JSON 資料 (已修改為支援多檔案)
        raw_data = self.load_pseudo_label_json(json_path)
        
        # 2. 預處理每個樣本
        processed_data = []
        # 使用 tqdm.auto.tqdm 以獲得更好的兼容性
        for i, example in enumerate(tqdm(raw_data, desc="正在預處理資料")):
            try:
                processed = self.preprocess_single_example(example)
                processed_data.append(processed)
            except Exception as e:
                # print(f"⚠ 處理第 {i} 筆資料 ({example.get('audio_path', 'N/A')}) 時發生錯誤: {e}")
                # 錯誤太多時會洗版，可以考慮只印出前幾個錯誤或計數
                continue
        
        # 3. 轉換為 Hugging Face Dataset
        dataset = Dataset.from_list(processed_data)
        
        print(f"\n✓ 成功處理 {len(dataset)} 筆資料 (已過濾 {len(raw_data) - len(dataset)} 筆)")
        try:
             print(f"Dataset 欄位: {dataset.column_names}")
        except:
            pass # 有時候空 dataset 會報錯
        
        # 4. 儲存處理後的資料集 (可選)
        if output_path:
            dataset.save_to_disk(output_path)
            print(f"✓ 資料集已儲存至: {output_path}")
        
        return dataset
    
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
            """自定義的 collate function 用於處理不同長度的序列"""
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
                'target_texts': [item['target_text'] for item in batch],
                'original_texts': [item['original_text'] for item in batch],
                'sentence_emotions': [item['sentence_emotion'] for item in batch],
                'audio_paths': [item['audio_path'] for item in batch]
            }
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    
    def visualize_example(self, example: Dict):
        """
        視覺化單一處理後的樣本，用於檢查
        """
        print("\n" + "="*80)
        print("樣本視覺化")
        print("="*80)
        print(f"音訊路徑: {example['audio_path']}")
        print(f"原始文字: {example['original_text']}")
        print(f"句子情緒: {example['sentence_emotion']}")
        print(f"\n目標序列 (文字): {example['target_text']}")
        
        print(f"\nLabels (token IDs): {example['labels'][:20]}...")
        print(f"Labels 長度: {len(example['labels'])}")
        
        print(f"\nDecoder Input IDs: {example['decoder_input_ids'][:20]}...")
        print(f"Decoder Input 長度: {len(example['decoder_input_ids'])}")
        
        try:
            features_shape = (len(example['input_features']), len(example['input_features'][0]) if example['input_features'] else 0)
        except Exception:
             # 如果已经是 numpy array
            features_shape = np.array(example['input_features']).shape

        print(f"\nInput Features shape: {features_shape}")
        print("="*80 + "\n")


# ============================================================================
# 使用範例
# ============================================================================

def main():
    """完整的資料預處理流程範例"""
    
    # 1. 初始化預處理器
    print("步驟 1: 初始化預處理器")
    preprocessor = WhisperEmotionDataPreprocessor(
        base_model="openai/whisper-large-v2",
        save_tokenizer_path="./custom_whisper_tokenizer"
    )
    
    # 2. 載入訓練集 (Sessions 2-5)
    print("\n步驟 2: 載入訓練集 (Sessions 2-5)")
    train_json_paths = [
        "/home/ouo/whisper_emotion/workspace/iemocap_processed/Session2.json",
        "/home/ouo/whisper_emotion/workspace/iemocap_processed/Session3.json",
        "/home/ouo/whisper_emotion/workspace/iemocap_processed/Session4.json",
        "/home/ouo/whisper_emotion/workspace/iemocap_processed/Session5.json"
    ]
    
    print("\n--- 載入並合併 Sessions 2-5 作為訓練集 ---")
    all_train_data = preprocessor.load_pseudo_label_json(train_json_paths)
    
    if len(all_train_data) == 0:
        print("❌ 錯誤: 訓練集為空!")
        return None, None, None
    
    print(f"✓ 訓練集總共載入 {len(all_train_data)} 筆資料")
    
    # 3. 分析訓練集情緒分布
    print("\n步驟 3: 分析訓練集情緒分布")
    from collections import Counter
    emotion_dist = Counter(sample['sentence_emotion'] for sample in all_train_data)
    
    print("訓練集情緒分布:")
    for emotion in ['neutral', 'happy', 'sad', 'angry']:
        count = emotion_dist[emotion]
        pct = count / len(all_train_data) * 100
        print(f"  {emotion:8s}: {count:4d} ({pct:5.1f}%)")
    
    # 4. 預處理訓練集
    print("\n步驟 4: 預處理訓練集")
    processed_train = []
    for example in tqdm(all_train_data, desc="處理訓練集"):
        try:
            processed = preprocessor.preprocess_single_example(example)
            processed_train.append(processed)
        except Exception as e:
            continue
    
    # 轉換為 Dataset
    from datasets import Dataset
    train_dataset = Dataset.from_list(processed_train)
    print(f"✓ 成功處理訓練集 {len(train_dataset)} 筆資料")
    
    # 儲存訓練集
    train_output_path = "./iemocap_processed/processed_train"
    train_dataset.save_to_disk(train_output_path)
    print(f"✓ 訓練集已儲存至: {train_output_path}")
    
    # 5. 載入並處理測試集 (Session 1)
    print("\n步驟 5: 載入並處理測試集 (Session 1)")
    test_json_path = "/home/ouo/whisper_emotion/workspace/iemocap_processed/iemocap_with_pseudo_labels_test_set.json"
    
    test_dataset = preprocessor.preprocess_dataset(
        json_path=test_json_path,
        output_path="./iemocap_processed/processed_test"
    )
    
    if len(test_dataset) == 0:
        print("❌ 警告: 測試集為空!")
        train_loader = preprocessor.create_dataloader(
            train_dataset,
            batch_size=4,
            shuffle=True
        )
        return preprocessor, train_loader, None
    
    # 分析測試集情緒分布
    test_data = preprocessor.load_pseudo_label_json(test_json_path)
    test_emotion_dist = Counter(sample['sentence_emotion'] for sample in test_data)
    
    print("\n測試集情緒分布:")
    for emotion in ['neutral', 'happy', 'sad', 'angry']:
        count = test_emotion_dist[emotion]
        pct = count / len(test_data) * 100 if len(test_data) > 0 else 0
        print(f"  {emotion:8s}: {count:4d} ({pct:5.1f}%)")
    
    # 6. 視覺化訓練集樣本
    print("\n步驟 6: 視覺化訓練集樣本")
    if len(train_dataset) > 0:
        preprocessor.visualize_example(train_dataset[0])
    
    # 7. 建立 DataLoader
    print("\n步驟 7: 建立 DataLoader")
    train_loader = preprocessor.create_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )
    
    test_loader = preprocessor.create_dataloader(
        test_dataset,
        batch_size=4,
        shuffle=False
    )
    
    # 8. 測試載入 batch
    print("\n步驟 8: 測試載入 batch")
    print("\n--- 訓練集 batch ---")
    for batch in train_loader:
        print(f"Input features shape: {batch['input_features'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Batch size: {len(batch['target_texts'])}")
        print(f"樣本情緒: {batch['sentence_emotions']}")
        break
    
    print("\n--- 測試集 batch ---")
    for batch in test_loader:
        print(f"Input features shape: {batch['input_features'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Batch size: {len(batch['target_texts'])}")
        print(f"樣本情緒: {batch['sentence_emotions']}")
        break
    
    # 9. 輸出最終統計
    print("\n" + "="*70)
    print("✓ 資料預處理完成！")
    print("="*70)
    print(f"訓練集 (Sessions 2-5): {len(train_dataset)} 樣本")
    print(f"  路徑: {train_output_path}")
    print(f"\n測試集 (Session 1): {len(test_dataset)} 樣本")
    print(f"  路徑: ./iemocap_processed/processed_test")
    print("="*70)
    
    print("\n資料集劃分:")
    print(f"  訓練集: Sessions 2, 3, 4, 5 → {len(train_dataset)} 樣本")
    print(f"  測試集: Session 1 → {len(test_dataset)} 樣本")
    print(f"  總計: {len(train_dataset) + len(test_dataset)} 樣本")
    
    return preprocessor, train_loader, test_loader


if __name__ == "__main__":
    result = main()
    if result and result[0] is not None:
        preprocessor, train_loader, test_loader = result
        print("\n✓ 成功! preprocessor, train_loader, test_loader 已準備好")
    else:
        print("\n❌ 預處理失敗!")
