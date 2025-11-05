"""
準備 Whisper 模型用於情緒識別訓練
優化版本：支持 BFloat16，減少 VRAM 使用
"""

import torch
from transformers import WhisperForConditionalGeneration, WhisperTokenizerFast, WhisperConfig
from pathlib import Path
import gc

# ============================================================================
# 配置
# ============================================================================

# 基礎模型（論文中使用 large-v2）
BASE_MODEL = "openai/whisper-large-v2"

# 自定義 tokenizer 路徑（在 data_preprocessor.py 中創建）
TOKENIZER_PATH = "./custom_whisper_tokenizer"

# 調整大小後模型的儲存路徑（必須與 train.py 中的 MODEL_PATH 一致）
RESIZED_MODEL_SAVE_PATH = "./my-whisper-emotion-model-resized"

# 使用 BFloat16（大幅減少 VRAM 使用）
USE_BFLOAT16 = True

# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" "*20 + "準備 Whisper 情緒識別模型")
    print("="*70)
    
    # === 步驟 1：載入 Tokenizer ===
    print(f"\n[1/4] 載入自定義 tokenizer from {TOKENIZER_PATH}...")
    try:
        tokenizer = WhisperTokenizerFast.from_pretrained(TOKENIZER_PATH)
        print(f"✓ Tokenizer 載入成功")
        print(f"  - 詞彙量: {len(tokenizer)}")
    except Exception as e:
        print(f"❌ 錯誤：無法載入 tokenizer - {e}")
        return
    
    # === 步驟 2：載入基礎模型 ===
    print(f"\n[2/4] 載入基礎模型 from {BASE_MODEL}...")
    
    if USE_BFLOAT16:
        print(f"  - 使用 BFloat16 精度（節省 ~50% VRAM）")
        try:
            model = WhisperForConditionalGeneration.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,  # 使用 BF16
                low_cpu_mem_usage=True       # 減少 CPU 記憶體使用
            )
            print(f"✓ 基礎模型 (bfloat16) 載入成功")
        except Exception as e:
            print(f"⚠️ BFloat16 載入失敗: {e}")
            print(f"  - 嘗試使用 Float32...")
            model = WhisperForConditionalGeneration.from_pretrained(
                BASE_MODEL,
                low_cpu_mem_usage=True
            )
            print(f"✓ 基礎模型 (float32) 載入成功")
    else:
        print(f"  - 使用 Float32 精度")
        model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            low_cpu_mem_usage=True
        )
        print(f"✓ 基礎模型 (float32) 載入成功")
    
    # 顯示模型資訊
    print(f"\n模型資訊:")
    print(f"  - 參數數量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  - 模型 dtype: {model.dtype}")
    
    # 清理記憶體
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  - GPU 記憶體使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # === 步驟 3：調整 Token Embeddings ===
    print(f"\n[3/4] 調整模型 Embedding 大小...")
    
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    new_vocab_size = len(tokenizer)
    
    print(f"  - 原始詞彙量: {original_vocab_size}")
    print(f"  - 新的詞彙量: {new_vocab_size}")
    print(f"  - 新增 tokens: {new_vocab_size - original_vocab_size}")
    
    if original_vocab_size != new_vocab_size:
        print(f"  - 正在調整 token embeddings...")
        
        # 調整 embedding 大小
        model.resize_token_embeddings(new_vocab_size)
        
        # 驗證調整結果
        new_shape = model.get_input_embeddings().weight.shape
        print(f"✓ Embedding 調整完成")
        print(f"  - 新的 embedding 形狀: {new_shape}")
        
        # 如果使用 BF16，確保新增的 embeddings 也是 BF16
        if USE_BFLOAT16:
            model = model.to(torch.bfloat16)
            print(f"  - 新增的 embeddings 已轉換為 bfloat16")
    else:
        print(f"✓ 詞彙量相同，無需調整 embeddings")
    
    # === 步驟 4：儲存模型 ===
    print(f"\n[4/4] 儲存調整後的模型...")
    
    save_path = Path(RESIZED_MODEL_SAVE_PATH)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 儲存模型權重
        print(f"  - 儲存模型權重到 {save_path}...")
        model.save_pretrained(
            save_path,
            safe_serialization=True  # 使用更安全的儲存格式
        )
        
        # 儲存配置
        print(f"  - 儲存模型配置...")
        config = model.config
        config.save_pretrained(save_path)
        
        # 儲存 tokenizer（可選，但建議一起儲存）
        print(f"  - 儲存 tokenizer...")
        tokenizer.save_pretrained(save_path)
        
        print(f"✓ 模型和配置已成功儲存至 {save_path}")
        
        # 顯示儲存的文件
        print(f"\n儲存的文件:")
        for file in sorted(save_path.glob("*")):
            size = file.stat().st_size / 1e6  # MB
            print(f"  - {file.name}: {size:.1f} MB")
        
    except Exception as e:
        print(f"❌ 儲存模型時發生錯誤: {e}")
        raise e
    
    # === 完成 ===
    print("\n" + "="*70)
    print("✅ 模型準備完成！")
    print("="*70)
    print(f"\n下一步：")
    print(f"  python train.py")
    print(f"\n訓練腳本將從 {save_path} 載入模型")
    
    # 顯示記憶體使用情況
    if torch.cuda.is_available():
        print(f"\nGPU 記憶體使用:")
        print(f"  - 已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  - 已保留: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"  - 總計: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
