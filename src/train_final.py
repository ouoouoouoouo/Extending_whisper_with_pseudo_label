"""
完整的訓練腳本 - Epoch 結束評估版
- 每個 epoch 結束後評估 WER, UAR, WA
- 使用 UAR 作為最佳模型判斷標準（適合不平衡資料）
- 使用 bf16 混合精度訓練
- 結果保存為 JSON 檔案
- 沒有驗證集 
"""

import torch
import torch.nn.functional as F
from transformers import WhisperTokenizerFast, WhisperConfig
from transformers import get_linear_schedule_with_warmup
from whisper_emotion_model import WhisperForEmotionRecognition
from generate_multitask_targets import WhisperEmotionDataPreprocessor
from preprocess_common_voice import CommonVoicePreprocessorForRehearsal

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import time
from pathlib import Path
from datasets import load_from_disk
from torch.amp import autocast
import itertools
import gc
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import evaluate
import json
from datetime import datetime

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"將使用 {device} 進行訓練...")

BASE_MODEL_NAME = "openai/whisper-large-v2"
TOKENIZER_PATH = "./custom_whisper_tokenizer"
MODEL_PATH = "./my-whisper-emotion-model-reinit-v2"

# ============================================================================
# 載入 Tokenizer
# ============================================================================
print(f"正在從 {TOKENIZER_PATH} 載入 Tokenizer...")
tokenizer = WhisperTokenizerFast.from_pretrained(TOKENIZER_PATH)

sle_token_ids_full = {e: id for e, id in tokenizer.get_added_vocab().items() if e.startswith("<|sle_")}
wle_token_ids_full = {e: id for e, id in tokenizer.get_added_vocab().items() if e.startswith("<|wle_")}

sle_token_ids = {k.replace("<|sle_", "").replace("|>", ""): v for k, v in sle_token_ids_full.items()}
wle_token_ids = {k.replace("<|wle_", "").replace("|>", ""): v for k, v in wle_token_ids_full.items()}

print(f"✓ SLE Token IDs: {sle_token_ids}")
print(f"✓ WLE Token IDs: {wle_token_ids}")

# 建立反向映射
id_to_emotion = {}
for emo, tid in sle_token_ids.items():
    id_to_emotion[tid] = emo
for emo, tid in wle_token_ids.items():
    id_to_emotion[tid] = emo

sle_ids_set = set(sle_token_ids.values())
wle_ids_set = set(wle_token_ids.values())

# ============================================================================
# 載入模型
# ============================================================================
print(f"\n正在從 {MODEL_PATH} 載入模型...")
config = WhisperConfig.from_pretrained(MODEL_PATH)

model = WhisperForEmotionRecognition.from_pretrained(
    MODEL_PATH,
    config=config,
    sle_token_ids=sle_token_ids,
    wle_token_ids=wle_token_ids,
    torch_dtype=torch.bfloat16  #用BF16
)

# 調整並正確初始化新增 token 的 embedding
new_num_tokens = len(tokenizer)
old_num_tokens = model.config.vocab_size

if new_num_tokens != old_num_tokens:
    model.resize_token_embeddings(new_num_tokens)
    if new_num_tokens > old_num_tokens:
        with torch.no_grad():
            print(f"🔧 初始化新增的 {new_num_tokens - old_num_tokens} 個 token embedding")
            model.model.decoder.embed_tokens.weight[old_num_tokens:new_num_tokens].normal_(mean=0.0, std=0.02)
            try:
                model.model.encoder.embed_tokens.weight[old_num_tokens:new_num_tokens].normal_(mean=0.0, std=0.02)
            except Exception:
                pass

model.config.vocab_size = new_num_tokens
model.to(device)
print("✓ 模型載入完成 (使用 bf16)")

# 配置 SAC mask
model.set_sac_mask_config(
    use_in_training=True,
    use_in_inference=False
)

print("\n✓ SAC Mask 配置:")
print("  - 訓練時: WLE 不能看到 SLE")
print("  - 推理時: 不使用 mask")

# ============================================================================
# 載入資料集
# ============================================================================
print("\n正在載入資料集...")

iemocap_preprocessor = WhisperEmotionDataPreprocessor(
    base_model=BASE_MODEL_NAME,
    save_tokenizer_path=TOKENIZER_PATH
)
iemocap_preprocessor.tokenizer = tokenizer

train_dataset_iemocap = load_from_disk("./iemocap_processed/processed_train")
# 改成載入 test set 作為驗證集（Session 1）
val_dataset_iemocap = load_from_disk("./iemocap_processed/processed_test")

cv_preprocessor = CommonVoicePreprocessorForRehearsal(
    custom_tokenizer_path=TOKENIZER_PATH,  
    base_model=BASE_MODEL_NAME,
    cv_data_path="/home/ouo/whisper_emotion/workspace/CV/en"  
)
cv_preprocessor.tokenizer = tokenizer

train_dataset_cv = load_from_disk("./cv_processed_for_rehearsal/train")
val_dataset_cv = load_from_disk("./cv_processed_for_rehearsal/val")
test_dataset_cv = load_from_disk("./cv_processed_for_rehearsal/test")

print(f"✓ IEMOCAP 訓練集: {len(train_dataset_iemocap)} 樣本 (Session 2-5)")
print(f"✓ IEMOCAP 驗證集: {len(val_dataset_iemocap)} 樣本 (Session 1)")
print(f"✓ Common Voice 訓練集: {len(train_dataset_cv)} 樣本")
print(f"✓ Common Voice 驗證集: {len(val_dataset_cv)} 樣本")
print(f"✓ Common Voice 測試集: {len(test_dataset_cv)} 樣本")

# ============================================================================
# 訓練參數
# ============================================================================
BATCH_SIZE = 2 # 論文使用 batch_size=4
ACCUMULATION_STEPS = 2 #有效batchsize=4
NUM_WORKERS = 4
WARMUP_STEPS = 0  # 論文未提及 warmup
LEARNING_RATE = 1e-5  # 論文未指定，保持預設
EPOCHS = 2  # 論文訓練 2 epochs
MODEL_SAVE_PATH = Path("./whisper_emotion_bf16_beta1")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

BETA = 1.0
VAL_BATCH_SIZE = 4  # 驗證時可以用較大的 batch

# 自動計算驗證頻率（1/32 epoch）
# 論文設定: batch_size=4
# IEMOCAP 約 4000 樣本，每個 epoch = 4000/4 = 1000 步
STEPS_PER_EPOCH = len(train_dataset_iemocap) // BATCH_SIZE
VALIDATION_STEPS = max(STEPS_PER_EPOCH // 32, 1)  # 1/32 epoch，最少 1 步

global_step_count = 0
best_uar = 0.0  # 改用 UAR 作為最佳模型判斷標準

print(f"\n驗證頻率計算:")
print(f"  預估每 epoch 步數: {STEPS_PER_EPOCH}")
print(f"  驗證間隔: 每 {VALIDATION_STEPS} 步 (1/32 epoch)")
print(f"  每 epoch 驗證次數: ~{STEPS_PER_EPOCH // VALIDATION_STEPS} 次")

print(f"\n訓練配置 (依據論文設定):")
print(f"  - 模型: whisper-large-v2 (論文用 large-v2)")
print(f"  - 批次大小: {BATCH_SIZE} (論文設定)")
print(f"  - 累積步數: {ACCUMULATION_STEPS}")
print(f"  - 學習率: {LEARNING_RATE}")
print(f"  - Epochs: {EPOCHS} (論文設定)")
print(f"  - Beta: {BETA}")
print(f"  - 驗證批次: {VAL_BATCH_SIZE}")
print(f"  - 模型儲存: {MODEL_SAVE_PATH}")
print(f"  - 精度: BF16")
print(f"  - 最佳模型標準: UAR (適合不平衡資料)")

# ============================================================================
# DataLoader
# ============================================================================
loader_iemocap = iemocap_preprocessor.create_dataloader(
    train_dataset_iemocap, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
loader_cv = cv_preprocessor.create_dataloader(
    train_dataset_cv, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

val_loader_iemocap = iemocap_preprocessor.create_dataloader(
    val_dataset_iemocap, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
val_loader_cv = cv_preprocessor.create_dataloader(
    val_dataset_cv, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader_cv = cv_preprocessor.create_dataloader(  # ← 改名稱
    test_dataset_cv, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# ============================================================================
# 優化器和排程器（依據論文）
# ============================================================================
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

# 論文使用 NewBobScheduler (ReduceLROnPlateau 的變體)
# 這裡使用標準的 ReduceLROnPlateau
plateau_scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='max',  # UAR 越高越好
    factor=0.5,  # LR 降低 50%
    patience=2,  # 2 次驗證無改善才降低
    #verbose=True,
    min_lr=1e-7  # 最小學習率
)

# 不使用 warmup scheduler (論文未提及)

#scaler = torch.cuda.amp.GradScaler()

# ============================================================================
# 快速驗證函數（只計算 loss 和 UAR）
# ============================================================================
def quick_validation(model, val_loader_iemocap, device, sle_token_ids, id_to_emotion, step):
    """
    快速驗證：只計算 loss 和 UAR（跳過 WER）
    用於訓練過程中的即時監控
    """
    model.eval()
    
    total_val_loss = 0
    val_batches = 0
    all_true_sle_ids = []
    all_pred_sle_ids = []
    
    sle_id_list = list(sle_token_ids.values())
    vocab_size = model.config.vocab_size
    ser_mask = torch.full((vocab_size,), float('-inf'), device=device)
    for token_id in sle_id_list:
        if token_id is not None and 0 <= token_id < vocab_size:
            ser_mask[token_id] = 0.0
    
    iemocap_prompt_len = 3
    iemocap_true_sle_idx = 3
    
    with torch.no_grad():
        # 只取部分樣本加速（最多 200 個樣本）
        val_samples = 0
        max_val_samples = 200
        
        for batch in val_loader_iemocap:
            if val_samples >= max_val_samples:
                break
                
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            true_decoder_ids = batch["decoder_input_ids"].to(device)
            batch_size = input_features.shape[0]
            
            # 計算 loss
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_features=input_features,
                    labels=labels,
                    decoder_input_ids=true_decoder_ids
                )
                total_val_loss += outputs.loss.item()
                val_batches += 1
            
            # 計算 SER
            true_sle_ids_batch = true_decoder_ids[:, iemocap_true_sle_idx].cpu().numpy()
            decoder_input_for_ser = true_decoder_ids[:, :iemocap_prompt_len]
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                ser_outputs = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_for_ser,
                    labels=None,
                )
                next_token_logits = ser_outputs.logits[:, -1, :]
                masked_logits = next_token_logits + ser_mask
                pred_sle_ids_batch = torch.argmax(masked_logits, dim=-1).cpu().numpy()
            
            all_true_sle_ids.extend(true_sle_ids_batch.tolist())
            all_pred_sle_ids.extend(pred_sle_ids_batch.tolist())
            
            val_samples += batch_size
    
    avg_val_loss = total_val_loss / max(val_batches, 1)
    
    # 計算 UAR
    uar = recall_score(
        all_true_sle_ids, all_pred_sle_ids,
        average='macro', labels=sle_id_list, zero_division=0
    )
    
    # 計算 WA
    wa = accuracy_score(all_true_sle_ids, all_pred_sle_ids)
    
    # 每類別召回率
    all_true_emotions = [id_to_emotion.get(id, 'unknown') for id in all_true_sle_ids]
    all_pred_emotions = [id_to_emotion.get(id, 'unknown') for id in all_pred_sle_ids]
    emotions = ['neutral', 'happy', 'sad', 'angry']
    cm = confusion_matrix(all_true_emotions, all_pred_emotions, labels=emotions)
    
    per_class_recall = {}
    for i, emotion in enumerate(emotions):
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        per_class_recall[emotion] = float(recall)
    
    print(f"\n{'='*70}")
    print(f"[Step {step}] 快速驗證（{val_samples} 樣本）")
    print(f"{'='*70}")
    print(f"  Loss: {avg_val_loss:.4f}")
    print(f"  UAR:  {uar * 100:.2f}%")
    print(f"  WA:   {wa * 100:.2f}%")
    print(f"  每類別召回率:")
    for emotion in emotions:
        recall_pct = per_class_recall[emotion] * 100
        print(f"    {emotion:8s}: {recall_pct:6.2f}%")
    print(f"{'='*70}\n")
    
    model.train()
    
    return avg_val_loss, uar, wa


# ============================================================================
# 評估函數
# ============================================================================
def evaluate_model(model, val_loader_iemocap, val_loader_cv, tokenizer, device, 
                   sle_token_ids, wle_token_ids, id_to_emotion, epoch):
    """
    完整評估函數：計算 WER, UAR, WA
    """
    print(f"\n{'='*70}")
    print(f"Epoch {epoch} 完整評估")
    print(f"{'='*70}")
    
    model.eval()
    
    # 準備評估指標
    wer_metric_iemocap = evaluate.load("wer")
    wer_metric_cv = evaluate.load("wer")
    
    all_true_sle_ids = []
    all_pred_sle_ids = []
    all_references_iemocap = []
    all_predictions_iemocap = []
    all_references_cv = []
    all_predictions_cv = []
    
    sle_id_list = list(sle_token_ids.values())
    wle_id_list = list(wle_token_ids.values())
    
    # 創建 SER 遮罩
    vocab_size = model.config.vocab_size
    ser_mask = torch.full((vocab_size,), float('-inf'), device=device)
    for token_id in sle_id_list:
        if token_id is not None and 0 <= token_id < vocab_size:
            ser_mask[token_id] = 0.0
    
    # IEMOCAP 格式參數
    iemocap_prompt_len = 3  # [SOT, EN, TRANSCRIBE]
    iemocap_true_sle_idx = 3
    
    # ========================================================================
    # 評估 IEMOCAP (SER + ASR)
    # ========================================================================
    print("\n評估 IEMOCAP...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader_iemocap, desc="IEMOCAP 驗證"):
            input_features = batch["input_features"].to(device)
            true_decoder_ids = batch["decoder_input_ids"].to(device)
            references = batch["original_texts"]
            batch_size = input_features.shape[0]
            
            # --- SER 評估 ---
            true_sle_ids_batch = true_decoder_ids[:, iemocap_true_sle_idx].cpu().numpy()
            decoder_input_for_ser = true_decoder_ids[:, :iemocap_prompt_len]
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_for_ser,
                    labels=None,
                )
                next_token_logits = outputs.logits[:, -1, :]
                masked_logits = next_token_logits + ser_mask
                pred_sle_ids_batch = torch.argmax(masked_logits, dim=-1).cpu().numpy()
            
            all_true_sle_ids.extend(true_sle_ids_batch.tolist())
            all_pred_sle_ids.extend(pred_sle_ids_batch.tolist())
            
            # --- ASR 評估 ---
            decoder_prompt_asr = true_decoder_ids[:, :4]  # [SOT, EN, TRANSCRIBE, SLE]
            suppress_token_list = wle_id_list
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted_ids_asr = model.generate(
                    input_features=input_features,
                    decoder_input_ids=decoder_prompt_asr,
                    max_length=448,
                    use_cache=False,
                    suppress_tokens=suppress_token_list,
                    num_beams=1,
                    do_sample=False,
                )
            
            predicted_ids_no_prompt = predicted_ids_asr[:, 4:]
            decoded_preds = tokenizer.batch_decode(
                predicted_ids_no_prompt, 
                skip_special_tokens=True
            )
            
            all_references_iemocap.extend(references)
            all_predictions_iemocap.extend(decoded_preds)
            wer_metric_iemocap.add_batch(predictions=decoded_preds, references=references)
    
    # ========================================================================
    # 評估 Common Voice (ASR)
    # ========================================================================
    print("\n評估 Common Voice...")
    
    sot_id = tokenizer.bos_token_id
    en_id = tokenizer.convert_tokens_to_ids("<|en|>")
    transcribe_id = tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notimestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    cv_prompt_ids = [sot_id, en_id, transcribe_id, notimestamps_id]
    cv_prompt_len = len(cv_prompt_ids)
    
    with torch.no_grad():
        for batch in tqdm(test_loader_cv, desc="Common Voice 測試"):
            input_features = batch["input_features"].to(device)
            references = batch["original_texts"]
            batch_size = input_features.shape[0]
            
            decoder_prompt_cv = torch.tensor([cv_prompt_ids] * batch_size).to(device)
            
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicted_ids_asr = model.generate(
                    input_features=input_features,
                    decoder_input_ids=decoder_prompt_cv,
                    max_length=448,
                    use_cache=True,
                )
            
            predicted_ids_no_prompt = predicted_ids_asr[:, cv_prompt_len:]
            decoded_preds = tokenizer.batch_decode(
                predicted_ids_no_prompt, 
                skip_special_tokens=True
            )
            
            all_references_cv.extend(references)
            all_predictions_cv.extend(decoded_preds)
            wer_metric_cv.add_batch(predictions=decoded_preds, references=references)
    
    # ========================================================================
    # 計算指標
    # ========================================================================
    
    # IEMOCAP 指標
    wer_iemocap = wer_metric_iemocap.compute()
    wa = accuracy_score(all_true_sle_ids, all_pred_sle_ids)
    uar = recall_score(
        all_true_sle_ids, all_pred_sle_ids,
        average='macro', labels=sle_id_list, zero_division=0
    )
    
    # 轉換為情緒標籤
    all_true_emotions = [id_to_emotion.get(id, 'unknown') for id in all_true_sle_ids]
    all_pred_emotions = [id_to_emotion.get(id, 'unknown') for id in all_pred_sle_ids]
    
    emotions = ['neutral', 'happy', 'sad', 'angry']
    cm = confusion_matrix(
        all_true_emotions, all_pred_emotions,
        labels=emotions
    )
    
    per_class_recall = {}
    for i, emotion in enumerate(emotions):
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        per_class_recall[emotion] = float(recall)
    
    # Common Voice 指標
    wer_cv = wer_metric_cv.compute()
    
    # ========================================================================
    # 顯示結果
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"Epoch {epoch} 評估結果")
    print(f"{'='*70}")
    
    print(f"\nIEMOCAP ({len(all_true_sle_ids)} 樣本):")
    print(f"  ASR - WER:  {wer_iemocap * 100:.2f}%")
    print(f"  SER - WA:   {wa * 100:.2f}%")
    print(f"  SER - UAR:  {uar * 100:.2f}% ⭐")
    
    print(f"\n  每類別召回率:")
    for emotion in emotions:
        recall_pct = per_class_recall[emotion] * 100
        print(f"    {emotion:8s}: {recall_pct:6.2f}%")
    
    print(f"\n  混淆矩陣:")
    print("           預測:")
    print("         neutral  happy    sad    angry")
    for i, emotion in enumerate(emotions):
        print(f"{emotion:8s}  {cm[i, 0]:5d}  {cm[i, 1]:5d}  {cm[i, 2]:5d}  {cm[i, 3]:5d}")
    
    print(f"\nCommon Voice ({len(all_references_cv)} 樣本):")
    print(f"  ASR - WER:  {wer_cv * 100:.2f}%")
    
    print(f"{'='*70}\n")
    
    model.train()
    
    return {
        'iemocap': {
            'wer': float(wer_iemocap),
            'wa': float(wa),
            'uar': float(uar),
            'per_class_recall': per_class_recall,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(all_true_sle_ids)
        },
        'common_voice': {
            'wer': float(wer_cv),
            'total_samples': len(all_references_cv)
        }
    }

# ============================================================================
# 訓練迴圈
# ============================================================================
print("\n" + "="*70)
print("開始訓練")
print("="*70 + "\n")

# 用於保存所有 epoch 的結果
training_history = []

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()
    
    total_train_loss = 0
    total_loss_iemocap = 0
    total_loss_cv = 0
    num_batches = 0
    
    # 建立訓練迭代器
    if len(loader_iemocap) > len(loader_cv):
        train_progress_bar = tqdm(loader_iemocap, desc=f"Epoch {epoch+1}/{EPOCHS}")
        cv_iter = itertools.cycle(loader_cv)
        main_loader = loader_iemocap
        is_cv_main = False
    else:
        train_progress_bar = tqdm(loader_cv, desc=f"Epoch {epoch+1}/{EPOCHS}")
        iemocap_iter = itertools.cycle(loader_iemocap)
        main_loader = loader_cv
        is_cv_main = True

    optimizer.zero_grad()
    
    for i, main_batch in enumerate(train_progress_bar):
        try:
            # 取得兩個資料集的批次
            if is_cv_main:
                batch_cv = main_batch
                batch_iemocap = next(iemocap_iter)
            else:
                batch_iemocap = main_batch
                batch_cv = next(cv_iter)

            # ======================================================
            # 1️⃣ IEMOCAP loss
            # ======================================================
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_iemocap = model(
                    input_features=batch_iemocap["input_features"].to(device),
                    labels=batch_iemocap["labels"].to(device),
                    decoder_input_ids=batch_iemocap["decoder_input_ids"].to(device)
                )
            loss_iemocap = outputs_iemocap.loss
            

            if torch.isnan(loss_iemocap) or torch.isinf(loss_iemocap):
                print(f"\n!! NaN/Inf Loss，跳過 !!")
                optimizer.zero_grad()
                continue
            (loss_iemocap / ACCUMULATION_STEPS).backward()

            # ======================================================
            # 2️⃣ CV loss
            # ======================================================
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs_cv = model(
                    input_features=batch_cv["input_features"].to(device),
                    labels=batch_cv["labels"].to(device),
                    decoder_input_ids=batch_cv["decoder_input_ids"].to(device)
                )
            loss_cv = outputs_cv.loss

            if torch.isnan(loss_cv) or torch.isinf(loss_cv):
                print(f"\n!! NaN/Inf CV Loss，跳過 !!")
                optimizer.zero_grad()
                continue
            (BETA * loss_cv / ACCUMULATION_STEPS).backward()

            # ======================================================
            # 3️⃣ 更新權重
            # ======================================================
            if (i + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step_count += 1  # ← 提前

                # ✅ 更新統計（loss 用 loss_iemocap + BETA * loss_cv）
                total_loss_batch = loss_iemocap.item() + (BETA * loss_cv.item())
                total_train_loss += total_loss_batch
                total_loss_iemocap += loss_iemocap.item()
                total_loss_cv += loss_cv.item()
                num_batches += 1

                train_progress_bar.set_postfix({
                    "iem": f"{loss_iemocap.item():.4f}",
                    "cv": f"{loss_cv.item():.4f}",
                    "total": f"{total_loss_batch:.4f}",
                })
                
                # ========================================================
                # 每 N 步快速驗證
                # ========================================================
                if global_step_count > 0 and (global_step_count % VALIDATION_STEPS == 0):
                    val_loss, val_uar, val_wa = quick_validation(
                        model, val_loader_iemocap, device, 
                        sle_token_ids, id_to_emotion, global_step_count
                    )
                    
                    # 如果 UAR 提升，保存檢查點
                    if val_uar > best_uar:
                        print(f"🎯 新的最佳 UAR: {val_uar * 100:.2f}% (step {global_step_count})")
                        print(f"   保存臨時檢查點...")
                        
                        checkpoint_dir = MODEL_SAVE_PATH / f"checkpoint_step_{global_step_count}"
                        checkpoint_dir.mkdir(exist_ok=True)
                        
                        model.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        
                        # 保存檢查點資訊
                        checkpoint_info = {
                            'step': global_step_count,
                            'epoch': epoch + 1,
                            'uar': float(val_uar),
                            'wa': float(val_wa),
                            'loss': float(val_loss),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        checkpoint_file = checkpoint_dir / "checkpoint_info.json"
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump(checkpoint_info, f, indent=2, ensure_ascii=False)
                        
                        print("✓ 檢查點已保存")
                    
                    gc.collect()
                    torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n訓練批次錯誤: {e}")
            import traceback
            traceback.print_exc()
            optimizer.zero_grad()
            continue

    # ========================================================================
    # Epoch 結束：計算訓練損失並進行完整評估
    # ========================================================================
    avg_train_loss = total_train_loss / max(num_batches, 1)
    avg_iem_loss = total_loss_iemocap / max(num_batches, 1)
    avg_cv_loss = total_loss_cv / max(num_batches, 1)
    epoch_time = (time.time() - epoch_start_time) / 60
    
    # 完整評估
    eval_results = evaluate_model(
        model, val_loader_iemocap, test_loader_cv, tokenizer, device,
        sle_token_ids, wle_token_ids, id_to_emotion, epoch + 1
    )
    
    current_uar = eval_results['iemocap']['uar']
    current_wa = eval_results['iemocap']['wa']
    current_wer_iem = eval_results['iemocap']['wer']
    current_wer_cv = eval_results['common_voice']['wer']
    
    # 更新學習率排程器（使用 UAR）
    plateau_scheduler.step(current_uar)
    
    # 保存 epoch 結果
    epoch_summary = {
        'epoch': epoch + 1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'training': {
            'total_loss': float(avg_train_loss),
            'iemocap_loss': float(avg_iem_loss),
            'cv_loss': float(avg_cv_loss),
            'time_minutes': float(epoch_time)
        },
        'validation': eval_results,
        'learning_rate': optimizer.param_groups[0]['lr']
    }
    training_history.append(epoch_summary)
    
    # 保存訓練歷史
    history_file = MODEL_SAVE_PATH / "training_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # ========================================================================
    # 根據 UAR 保存最佳模型
    # ========================================================================
    if current_uar > best_uar:
        best_uar = current_uar
        
        print(f"\n🎯 新的最佳 UAR: {best_uar * 100:.2f}%！")
        print(f"   保存模型至 {MODEL_SAVE_PATH}...")
        
        model.save_pretrained(MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MODEL_SAVE_PATH)
        
        # 保存最佳模型資訊
        best_model_info = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'uar': float(best_uar),
            'wa': float(current_wa),
            'wer_iemocap': float(current_wer_iem),
            'wer_cv': float(current_wer_cv),
            'per_class_recall': eval_results['iemocap']['per_class_recall'],
            'confusion_matrix': eval_results['iemocap']['confusion_matrix']
        }
        
        best_model_file = MODEL_SAVE_PATH / "best_model_info.json"
        with open(best_model_file, 'w', encoding='utf-8') as f:
            json.dump(best_model_info, f, indent=2, ensure_ascii=False)
        
        print("✓ 模型已保存")
    
    # ========================================================================
    # Epoch 總結
    # ========================================================================
    print("\n" + "="*70)
    print(f"Epoch {epoch+1}/{EPOCHS} 完成 ({epoch_time:.1f} 分鐘)")
    print("="*70)
    print(f"訓練損失:")
    print(f"  總 Loss:      {avg_train_loss:.4f}")
    print(f"  IEMOCAP Loss: {avg_iem_loss:.4f}")
    print(f"  CV Loss:      {avg_cv_loss:.4f}")
    # 在 epoch 結束時顯示
    print(f"  L_IEM / L_CV 比例: {avg_iem_loss / max(avg_cv_loss, 1e-8):.2f}")  
    print(f"\n驗證結果:")
    print(f"  UAR:          {current_uar * 100:.2f}% {'⭐ (最佳)' if current_uar == best_uar else ''}")
    print(f"  WA:           {current_wa * 100:.2f}%")
    print(f"  WER (IEMOCAP): {current_wer_iem * 100:.2f}%")
    print(f"  WER (CV):     {current_wer_cv * 100:.2f}%")
    print(f"\n最佳 UAR:      {best_uar * 100:.2f}%")
    print("="*70 + "\n")
    
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# 訓練完成
# ============================================================================
print("\n" + "="*70)
print("🎉 訓練完成！")
print("="*70)
print(f"最佳模型已保存至: {MODEL_SAVE_PATH}")
print(f"最佳 UAR: {best_uar * 100:.2f}%")
print(f"訓練歷史已保存至: {MODEL_SAVE_PATH / 'training_history.json'}")
print(f"最佳模型資訊已保存至: {MODEL_SAVE_PATH / 'best_model_info.json'}")
print("="*70)
