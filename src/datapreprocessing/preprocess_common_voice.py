import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import WhisperTokenizer, WhisperFeatureExtractor
from datasets import Dataset, Audio, load_dataset, IterableDataset
import numpy as np
from tqdm.auto import tqdm

# ============================================================================
# V4: 可直接下載 (使用 fsicoli/common_voice_15_0)
# ============================================================================

class WhisperAudioProcessor:
    """處理音訊：接收音訊陣列、截斷、填充"""
    
    def __init__(self, feature_extractor: WhisperFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.sampling_rate = 16000
        self.max_audio_length_seconds = 30
        self.max_audio_samples = self.max_audio_length_seconds * self.sampling_rate
        self.max_feature_length = 3000 
        print(f"✓ (CV) 音訊處理器：音訊將被截斷/填充至 {self.max_audio_length_seconds} 秒")

    def process_audio_array(self, audio_array: np.ndarray) -> Dict[str, np.ndarray]:
        try:
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array)
            if len(audio_array) > self.max_audio_samples:
                audio_array = audio_array[:self.max_audio_samples]
            
            input_features = self.feature_extractor(
                audio_array,
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
            print(f"⚠ 處理音訊陣列失敗: {e}")
            raise e


class CommonVoicePreprocessor:
    """用於處理 Common Voice 15.0 (fsicoli 版本，可直接下載)"""
    
    def __init__(self, base_model: str = "openai/whisper-base", lang_token: str = "<|en|>"):
        print("="*80)
        print("初始化 Common Voice 15.0 (開放版本) 預處理器")
        print("="*80)
        
        self.tokenizer = WhisperTokenizer.from_pretrained(base_model)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
        self.audio_processor = WhisperAudioProcessor(self.feature_extractor)
        
        self.prefix_ids = [
            self.tokenizer.bos_token_id,
            self.tokenizer.convert_tokens_to_ids(lang_token),
            self.tokenizer.convert_tokens_to_ids("<|transcribe|>"),
            self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        ]
        
    def preprocess_hf_example(self, example: Dict) -> Optional[Dict]:
        try:
            audio_data = example['audio']
            transcript = example.get('sentence', None)
            if not transcript or len(transcript.strip()) == 0:
                return None

            audio_features = self.audio_processor.process_audio_array(audio_data['array'])
            custom_ids = self.tokenizer.encode(transcript, add_special_tokens=False)
            full_sequence = self.prefix_ids + custom_ids + [self.tokenizer.eos_token_id]
            
            decoder_input_ids = full_sequence[:-1]
            labels = full_sequence[1:]
            
            labels_with_prefix_ignored = labels.copy()
            for i in range(min(3, len(labels_with_prefix_ignored))):
                labels_with_prefix_ignored[i] = -100

            return {
                'input_features': audio_features['input_features'],
                'labels': labels_with_prefix_ignored,
                'decoder_input_ids': decoder_input_ids,
                'original_text': transcript
            }
        except Exception:
            return None

    def stream_and_process(self, hf_dataset: IterableDataset, num_samples: int, output_path: str, split_name: str):
        print(f"\n開始處理 {split_name} (目標: {num_samples} 筆)...")
        processed_data = []
        iterator = iter(hf_dataset)

        with tqdm(total=num_samples, desc=f"下載並處理 {split_name}") as pbar:
            while len(processed_data) < num_samples:
                try:
                    example = next(iterator)
                    processed = self.preprocess_hf_example(example)
                    if processed is not None:
                        processed_data.append(processed)
                        pbar.update(1)
                except StopIteration:
                    print(f"⚠ 警告: 資料集用盡，只收集到 {len(processed_data)} 筆資料")
                    break
                except Exception:
                    continue

        print(f"正在儲存 {split_name} 至磁碟...")
        dataset = Dataset.from_list(processed_data)
        dataset.save_to_disk(output_path)
        print(f"✓ {split_name} 已儲存至: {output_path} (共 {len(dataset)} 筆)")


def main():
    # 選擇開放可下載的版本
    HF_DATASET_NAME = "fsicoli/common_voice_15_0"
    LANG = "en"  # 可改成 "zh-TW"
    
    # 選擇語言 token
    lang_token = "<|en|>" if LANG == "en" else "<|zh|>"
    preprocessor = CommonVoicePreprocessor(base_model="openai/whisper-base", lang_token=lang_token)

    print(f"\n準備連線至 Hugging Face 下載 {HF_DATASET_NAME} [{LANG}]...")

    try:
        cv_stream = load_dataset(
            HF_DATASET_NAME,
            LANG,
            split="train",
            streaming=True
        )
        cv_stream = cv_stream.cast_column("audio", Audio(sampling_rate=16000))
        print("✓ 成功連線資料集，設定串流緩衝區...")
        cv_stream = cv_stream.shuffle(seed=42, buffer_size=10000)

    except Exception as e:
        print(f"\n❌ 無法載入 Hugging Face 資料集: {e}")
        print("請確認網路連線或 Hugging Face 是否可訪問。")
        return

    SAMPLES_PER_HOUR = int(3600 / 5)
    TRAIN_SAMPLES = SAMPLES_PER_HOUR * 10
    VAL_SAMPLES = SAMPLES_PER_HOUR * 1
    TEST_SAMPLES = SAMPLES_PER_HOUR * 1

    preprocessor.stream_and_process(cv_stream.take(TEST_SAMPLES), TEST_SAMPLES, "./cv15_processed/test", "CV15 Test")
    preprocessor.stream_and_process(cv_stream.skip(TEST_SAMPLES).take(VAL_SAMPLES), VAL_SAMPLES, "./cv15_processed/val", "CV15 Val")
    preprocessor.stream_and_process(cv_stream.skip(TEST_SAMPLES + VAL_SAMPLES).take(TRAIN_SAMPLES), TRAIN_SAMPLES, "./cv15_processed/train", "CV15 Train")

    print("\n✓ Common Voice 15.0 (fsicoli) 預處理全部完成！")


if __name__ == "__main__":
    main()
