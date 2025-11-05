import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, Audio
import librosa
import numpy as np # <-- 確保導入 numpy


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
        
        # --- **新增：定義最大音訊長度** ---
        self.max_audio_length_seconds = 30
        self.sampling_rate = 16000
        self.max_audio_samples = self.max_audio_length_seconds * self.sampling_rate
        print(f"✓ 音訊將被截斷至 {self.max_audio_length_seconds} 秒 ({self.max_audio_samples} 個取樣點)")

    
    def load_pseudo_label_json(self, json_path: str) -> List[Dict]:
        """
        載入包含 pseudo label 的 JSON 檔案
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✓ 載入 {len(data)} 筆資料從 {json_path}")
        return data
    
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
        處理音訊檔案，轉換為 Whisper 所需的特徵
        ** (已加入截斷和填充邏輯) **
        """
        # 載入音訊 (Whisper 使用 16kHz 採樣率)
        audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
        
        # --- **關鍵修正：截斷過長的音訊** ---
        if len(audio) > self.max_audio_samples:
            audio = audio[:self.max_audio_samples]
            
        # 使用 feature extractor 轉換為 log-mel spectrogram
        # return_tensors="pt" 會返回 PyTorch 張量，我們改為 "np" 以便儲存
        input_features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="np" # 使用 numpy array
        ).input_features
        
        # input_features shape (1, 80, 3000) -> (80, 3000)
        input_features = input_features.squeeze(0) 

        # --- **關鍵修正：填充過短的音訊** ---
        # Whisper 期望特徵長度為 3000
        feature_length = input_features.shape[1]
        if feature_length < 3000:
            # 創建一個全 0 (log-mel 中的 0) 的 numpy array 用於填充
            padding = np.zeros((input_features.shape[0], 3000 - feature_length), dtype=input_features.dtype)
            input_features = np.concatenate([input_features, padding], axis=1)

        return {
            "input_features": input_features 
        }
    
    def preprocess_single_example(self, example: Dict) -> Dict:
        """
        預處理單一樣本
        
        論文中的序列格式（Format C）：
        <|startoftranscript|> <|en|> <|transcribe|> <|sle_emotion|> word1 <|wle_e1|> word2 ... <|endoftext|>
        
        訓練時：
        - decoder_input_ids: 完整序列去掉最後的 EOS（用於 teacher forcing）
        - labels: 完整序列去掉第一個 token（用於計算 loss，已 shift）
        """
        # 1. 處理音訊
        audio_features = self.process_audio(example['audio_path'])
        
        # 2. 建立目標序列
        target_text, target_ids = self.create_target_sequence(
            transcript=example['transcript'],
            sentence_emotion=example['sentence_emotion'],
            word_level_emotions=example['word_level_emotions']
        )
        
        # target_ids 格式：
        # [<BOS>, <lang>, <task>, <sle>, word1, <wle1>, ..., <EOS>]
        
        # 3. 建立 decoder_input_ids（去掉最後的 EOS）
        decoder_input_ids = target_ids[:-1]
        # [<BOS>, <lang>, <task>, <sle>, word1, <wle1>, ...]
        
        # 4. 建立 labels（去掉第一個 BOS，保留 EOS）
        labels = target_ids[1:]
        # [<lang>, <task>, <sle>, word1, <wle1>, ..., <EOS>]
        
        # 5. (可選) 將前綴 tokens 設為 -100，不計算 loss
        # 這樣模型只學習預測情緒和文字，不學習預測語言/任務 token
        labels_with_prefix_ignored = labels.copy()
        labels_with_prefix_ignored[0] = -100  # <lang>
        labels_with_prefix_ignored[1] = -100  # <task>
        # 不要 mask <sle>，因為這是我們要學習的！
        
        return {
            'input_features': audio_features['input_features'],
            'labels': labels_with_prefix_ignored,  # ← 已經 shift，且 mask 了前綴
            'decoder_input_ids': decoder_input_ids,
            'target_text': target_text,
            'original_text': example['transcript'],
            'sentence_emotion': example['sentence_emotion'],
            'audio_path': example['audio_path']
        }
    
    def preprocess_dataset(
        self,
        json_path: str,
        output_path: str = None
    ) -> Dataset:
        """
        預處理整個資料集
        """
        # 1. 載入 JSON 資料
        raw_data = self.load_pseudo_label_json(json_path)
        
        # 2. 預處理每個樣本
        processed_data = []
        for i, example in enumerate(tqdm(raw_data, desc="正在預處理資料")):
            try:
                processed = self.preprocess_single_example(example)
                processed_data.append(processed)
            except Exception as e:
                print(f"⚠ 處理第 {i} 筆資料 ({example.get('audio_path', 'N/A')}) 時發生錯誤: {e}")
                continue
        
        # 3. 轉換為 Hugging Face Dataset
        dataset = Dataset.from_list(processed_data)
        
        print(f"\n✓ 成功處理 {len(dataset)} 筆資料 (已過濾 {len(raw_data) - len(dataset)} 筆)")
        print(f"Dataset 欄位: {dataset.column_names}")
        
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
            # 取得 batch 中最長的序列長度
            max_label_length = max(len(item['labels']) for item in batch)
            max_decoder_length = max(len(item['decoder_input_ids']) for item in batch)
            
            # input_features 現在都是 30 秒 (80, 3000)，可以直接堆疊
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
                'target_texts': [item['target_text'] for item in batch],
                'original_texts': [item['original_text'] for item in batch],
                'sentence_emotions': [item['sentence_emotion'] for item in batch],
                'audio_paths': [item['audio_path'] for item in batch] # 新增 (用於 debug)
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
        
        # input_features 是 list of lists, 手動計算 shape
        try:
            features_shape = (len(example['input_features']), len(example['input_features'][0]) if example['input_features'] else 0)
        except Exception:
            features_shape = (len(example['input_features']),)
        
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
    
    # 2. 預處理資料集
    print("\n步驟 2: 預處理資料集")
    train_dataset = preprocessor.preprocess_dataset(
        json_path="/home/ouo/whisper_emotion/workspace/iemocap_processed/iemocap_with_pseudo_labels_train_set.json",
        output_path="./iemocap_processed/processed_train"
    )
    
    val_dataset = preprocessor.preprocess_dataset(
        json_path="/home/ouo/whisper_emotion/workspace/iemocap_processed/iemocap_with_pseudo_labels_validation_set.json",
        output_path="./iemocap_processed/processed_val"
    )
    test_dataset=preprocessor.preprocess_dataset(
        json_path="/home/ouo/whisper_emotion/workspace/iemocap_processed/iemocap_with_pseudo_labels_test_set.json",
        output_path="./iemocap_processed/processed_test"
    )
    
    # 3. 視覺化一個樣本
    print("\n步驟 3: 視覺化樣本")
    preprocessor.visualize_example(train_dataset[0])
    
    # 4. 建立 DataLoader
    print("\n步驟 4: 建立 DataLoader")
    train_loader = preprocessor.create_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=True
    )
    
    val_loader = preprocessor.create_dataloader(
        val_dataset,
        batch_size=4,
        shuffle=False
    )
    test_loader=preprocessor.create_dataloader(
        test_dataset, # <-- 修正：使用 test_dataset
        batch_size=4,
        shuffle=False # <-- 修正：測試時不應打亂
    )
    # 5. 測試載入一個 batch
    print("\n步驟 5: 測試載入 batch")
    for batch in train_loader:
        print(f"Batch input_features shape: {batch['input_features'].shape}")
        print(f"Batch labels shape: {batch['labels'].shape}")
        print(f"Batch decoder_input_ids shape: {batch['decoder_input_ids'].shape}")
        print(f"Batch size: {len(batch['target_texts'])}")
        print(f"第一個樣本的目標文字: {batch['target_texts'][0]}")
        break
    
    print("\n✓ 資料預處理完成！")
    
    return preprocessor, train_loader, val_loader, test_loader




if __name__ == "__main__":
    from tqdm.auto import tqdm # 將 tqdm 導入到主範圍
    preprocessor, train_loader, val_loader, test_loader = main()

