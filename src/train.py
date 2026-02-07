"""
完整的訓練腳本 - Beta 掃描版
- 測試 beta 從 0.0 到 1.5（每次增加 0.1）
- 每個 beta 值訓練 2 epochs
- 記錄每個 beta 的最佳 UAR 和對應指標
- 最終生成完整的 beta 掃描結果 JSON
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

import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



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
# 載入資料集
# ============================================================================
print("\n正在載入資料集...")

iemocap_preprocessor = WhisperEmotionDataPreprocessor(
    base_model=BASE_MODEL_NAME,
    save_tokenizer_path=TOKENIZER_PATH
)
iemocap_preprocessor.tokenizer = tokenizer

train_dataset_iemocap = load_from_disk("./iemocap_processed/processed_train")
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
BATCH_SIZE = 2
ACCUMULATION_STEPS = 2
NUM_WORKERS = 4
WARMUP_STEPS = 0
LEARNING_RATE = 1e-5
EPOCHS = 2
BASE_SAVE_DIR = Path("./whisper_emotion_bf16_beta_scan")
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Beta 掃描範圍：0.0 到 1.5，每次增加 0.1
BETA_VALUES = [round(b, 2) for b in np.arange(0.3, 0.6, 0.02)]
VAL_BATCH_SIZE = 4

# 自動計算驗證頻率（1/32 epoch）
STEPS_PER_EPOCH = len(train_dataset_iemocap) // BATCH_SIZE
VALIDATION_STEPS = max(STEPS_PER_EPOCH // 32, 1)

print(f"\n訓練配置:")
print(f"  - 模型: whisper-large-v2")
print(f"  - 批次大小: {BATCH_SIZE}")
print(f"  - 學習率: {LEARNING_RATE}")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Beta 掃描範圍: {BETA_VALUES[0]} ~ {BETA_VALUES[-1]} (共 {len(BETA_VALUES)} 個值)")
print(f"  - 基礎儲存目錄: {BASE_SAVE_DIR}")
print(f"  - 精度: BF16")

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
b = next(iter(val_loader_iemocap))
print("decoder_input_ids[0][:10] =", b["decoder_input_ids"][0][:10].tolist())
print("decoded =", tokenizer.decode(b["decoder_input_ids"][0], skip_special_tokens=False))
print("idx3 token =", b["decoder_input_ids"][0][3].item(), tokenizer.decode([b["decoder_input_ids"][0][3].item()]))

val_loader_cv = cv_preprocessor.create_dataloader(
    val_dataset_cv, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader_cv = cv_preprocessor.create_dataloader(
    test_dataset_cv, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

# ============================================================================
# 評估函數
# ============================================================================
def evaluate_model(model, val_loader_iemocap, val_loader_cv, tokenizer, device, 
                   sle_token_ids, wle_token_ids, id_to_emotion, epoch, beta):
    """完整評估函數：計算 WER, UAR, WA"""
    print(f"\n{'='*70}")
    print(f"Beta={beta:.2f}, Epoch {epoch} 完整評估")
    print(f"{'='*70}")
    
    model.eval()
    
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
    
    vocab_size = model.config.vocab_size
    ser_mask = torch.full((vocab_size,), float('-inf'), device=device)
    for token_id in sle_id_list:
        if token_id is not None and 0 <= token_id < vocab_size:
            ser_mask[token_id] = 0.0
    
    iemocap_prompt_len = 3
    iemocap_true_sle_idx = 3
    
    # 評估 IEMOCAP
    print("\n評估 IEMOCAP...")
    with torch.no_grad():
        for batch in tqdm(val_loader_iemocap, desc="IEMOCAP 驗證"):
            input_features = batch["input_features"].to(device)
            true_decoder_ids = batch["decoder_input_ids"].to(device)
            references = batch["original_texts"]
            
            # SER
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
            
            # ASR
            decoder_prompt_asr = true_decoder_ids[:, :4]
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
    
    # 評估 Common Voice
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
    
    # 計算指標
    wer_iemocap = wer_metric_iemocap.compute()
    wa = accuracy_score(all_true_sle_ids, all_pred_sle_ids)
    uar = recall_score(
        all_true_sle_ids, all_pred_sle_ids,
        average='macro', labels=sle_id_list, zero_division=0
    )
    
    all_true_emotions = [id_to_emotion.get(id, 'unknown') for id in all_true_sle_ids]
    all_pred_emotions = [id_to_emotion.get(id, 'unknown') for id in all_pred_sle_ids]
    
    emotions = ['neutral', 'happy', 'sad', 'angry']
    cm = confusion_matrix(all_true_emotions, all_pred_emotions, labels=emotions)
    
    per_class_recall = {}
    for i, emotion in enumerate(emotions):
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        per_class_recall[emotion] = float(recall)
    
    wer_cv = wer_metric_cv.compute()
    
    # 顯示結果
    print(f"\n{'='*70}")
    print(f"Beta={beta:.2f}, Epoch {epoch} 評估結果")
    print(f"{'='*70}")
    
    print(f"\nIEMOCAP ({len(all_true_sle_ids)} 樣本):")
    print(f"  ASR - WER:  {wer_iemocap * 100:.2f}%")
    print(f"  SER - WA:   {wa * 100:.2f}%")
    print(f"  SER - UAR:  {uar * 100:.2f}% ⭐")
    
    print(f"\n  每類別召回率:")
    for emotion in emotions:
        recall_pct = per_class_recall[emotion] * 100
        print(f"    {emotion:8s}: {recall_pct:6.2f}%")
    
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
# Beta 掃描主迴圈
# ============================================================================
print("\n" + "="*70)
print("開始 Beta 掃描訓練")
print("="*70 + "\n")

# 儲存所有 beta 的結果
all_beta_results = []

for beta_idx, BETA in enumerate(BETA_VALUES):
    # ✅ 每個 beta 都重新設定 seed
    set_seed(42)
    
   
    print("\n" + "="*70)
    print(f"🔍 Beta 掃描 [{beta_idx + 1}/{len(BETA_VALUES)}]: Beta = {BETA:.1f}")
    print("="*70 + "\n")
    
    # 為每個 beta 創建獨立的保存路徑
    MODEL_SAVE_PATH = BASE_SAVE_DIR / f"beta_{BETA:.2f}"
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    # ✅ 重新建立 DataLoader
    print("正在建立 DataLoader...")
    loader_iemocap = iemocap_preprocessor.create_dataloader(
        train_dataset_iemocap, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    loader_cv = cv_preprocessor.create_dataloader(
        train_dataset_cv, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    # 重新載入模型（每次 beta 都從初始狀態開始）
    print(f"正在從 {MODEL_PATH} 載入新模型...")
    config = WhisperConfig.from_pretrained(MODEL_PATH)
    
    model = WhisperForEmotionRecognition.from_pretrained(
        MODEL_PATH,
        config=config,
        sle_token_ids=sle_token_ids,
        wle_token_ids=wle_token_ids,
        torch_dtype=torch.bfloat16
    )
    
    # 調整 token embeddings
    new_num_tokens = len(tokenizer)
    old_num_tokens = model.config.vocab_size
    
    if new_num_tokens != old_num_tokens:
        model.resize_token_embeddings(new_num_tokens)
        if new_num_tokens > old_num_tokens:
            with torch.no_grad():
                model.model.decoder.embed_tokens.weight[old_num_tokens:new_num_tokens].normal_(mean=0.0, std=0.02)
                try:
                    model.model.encoder.embed_tokens.weight[old_num_tokens:new_num_tokens].normal_(mean=0.0, std=0.02)
                except Exception:
                    pass
    
    model.config.vocab_size = new_num_tokens
    model.to(device)
    
    model.set_sac_mask_config(use_in_training=True, use_in_inference=False)
    
    # 重新初始化優化器和排程器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
        threshold=0.0025, threshold_mode='rel', min_lr=1e-7
    )
    
    # 訓練變數
    global_step_count = 0
    best_uar = 0.0
    training_history = []
    
    print(f"✓ 模型已載入並初始化 (Beta={BETA:.2f})")
    
    # ========================================================================
    # 訓練迴圈
    # ========================================================================
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        
        total_train_loss = 0
        total_loss_iemocap = 0
        total_loss_cv = 0
        num_batches = 0
        
        # 建立訓練迭代器
        if len(loader_iemocap) > len(loader_cv):
            train_progress_bar = tqdm(loader_iemocap, desc=f"Beta={BETA:.2f}, Epoch {epoch+1}/{EPOCHS}")
            cv_iter = itertools.cycle(loader_cv)
            main_loader = loader_iemocap
            is_cv_main = False
        else:
            train_progress_bar = tqdm(loader_cv, desc=f"Beta={BETA:.2f}, Epoch {epoch+1}/{EPOCHS}")
            iemocap_iter = itertools.cycle(loader_iemocap)
            main_loader = loader_cv
            is_cv_main = True
        
        optimizer.zero_grad()
        
        for i, main_batch in enumerate(train_progress_bar):
            try:
                if is_cv_main:
                    batch_cv = main_batch
                    batch_iemocap = next(iemocap_iter)
                else:
                    batch_iemocap = main_batch
                    batch_cv = next(cv_iter)
                
                # IEMOCAP loss
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
                
                # CV loss
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
                
                # 更新權重
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step_count += 1
                    
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
                
            except Exception as e:
                print(f"\n訓練批次錯誤: {e}")
                import traceback
                traceback.print_exc()
                optimizer.zero_grad()
                continue
        
        # Epoch 結束評估
        avg_train_loss = total_train_loss / max(num_batches, 1)
        avg_iem_loss = total_loss_iemocap / max(num_batches, 1)
        avg_cv_loss = total_loss_cv / max(num_batches, 1)
        epoch_time = (time.time() - epoch_start_time) / 60
        
        eval_results = evaluate_model(
            model, val_loader_iemocap, test_loader_cv, tokenizer, device,
            sle_token_ids, wle_token_ids, id_to_emotion, epoch + 1, BETA
        )
        
        current_uar = eval_results['iemocap']['uar']
        current_wa = eval_results['iemocap']['wa']
        current_wer_iem = eval_results['iemocap']['wer']
        current_wer_cv = eval_results['common_voice']['wer']
        
        scheduler.step(current_uar)
        
        # 保存 epoch 結果
        epoch_summary = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training': {
                'total_loss': float(avg_train_loss),
                'iemocap_loss': float(avg_iem_loss),
                'cv_loss': float(avg_cv_loss),
                'loss_ratio': float(avg_iem_loss / max(avg_cv_loss, 1e-8)),
                'time_minutes': float(epoch_time)
            },
            'validation': eval_results,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_summary)
        
        # 如果是最佳 UAR，保存模型
        if current_uar > best_uar:
            best_uar = current_uar
            
            print(f"\n🎯 新的最佳 UAR: {best_uar * 100:.2f}%！")
            print(f"   保存模型至 {MODEL_SAVE_PATH}...")
            
            model.save_pretrained(MODEL_SAVE_PATH)
            tokenizer.save_pretrained(MODEL_SAVE_PATH)
            
            best_model_info = {
                'beta': float(BETA),
                'epoch': epoch + 1,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'uar': float(best_uar),
                'wa': float(current_wa),
                'wer_iemocap': float(current_wer_iem),
                'wer_cv': float(current_wer_cv),
                'per_class_recall': eval_results['iemocap']['per_class_recall'],
                'confusion_matrix': eval_results['iemocap']['confusion_matrix'],
                'training_loss': {
                    'iemocap': float(avg_iem_loss),
                    'cv': float(avg_cv_loss),
                    'ratio': float(avg_iem_loss / max(avg_cv_loss, 1e-8))
                }
            }
            
            best_model_file = MODEL_SAVE_PATH / "best_model_info.json"
            with open(best_model_file, 'w', encoding='utf-8') as f:
                json.dump(best_model_info, f, indent=2, ensure_ascii=False)
            
            print("✓ 模型已保存")
        
        # Epoch 總結
        print("\n" + "="*70)
        print(f"Beta={BETA:.2f}, Epoch {epoch+1}/{EPOCHS} 完成 ({epoch_time:.1f} 分鐘)")
        print("="*70)
        print(f"訓練損失:")
        print(f"  總 Loss:      {avg_train_loss:.4f}")
        print(f"  IEMOCAP Loss: {avg_iem_loss:.4f}")
        print(f"  CV Loss:      {avg_cv_loss:.4f}")
        print(f"  L_IEM / L_CV: {avg_iem_loss / max(avg_cv_loss, 1e-8):.2f}")
        print(f"\n驗證結果:")
        print(f"  UAR:          {current_uar * 100:.2f}% {'⭐ (最佳)' if current_uar == best_uar else ''}")
        print(f"  WA:           {current_wa * 100:.2f}%")
        print(f"  WER (IEMOCAP): {current_wer_iem * 100:.2f}%")
        print(f"  WER (CV):     {current_wer_cv * 100:.2f}%")
        print(f"\n最佳 UAR:      {best_uar * 100:.2f}%")
        print("="*70 + "\n")
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # ========================================================================
    # 保存此 beta 的訓練歷史
    # ========================================================================
    history_file = MODEL_SAVE_PATH / "training_history.json"
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    
    # 記錄此 beta 的最佳結果
    beta_summary = {
        'beta': float(BETA),
        'best_uar': float(best_uar),
        'model_path': str(MODEL_SAVE_PATH),
        'completed_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 如果有 best_model_info，加入更多細節
    best_info_file = MODEL_SAVE_PATH / "best_model_info.json"
    if best_info_file.exists():
        with open(best_info_file, 'r', encoding='utf-8') as f:
            best_info = json.load(f)
            beta_summary.update({
                'best_epoch': best_info['epoch'],
                'wa': best_info['wa'],
                'wer_iemocap': best_info['wer_iemocap'],
                'wer_cv': best_info['wer_cv'],
                'per_class_recall': best_info['per_class_recall'],
                'training_loss': best_info['training_loss']
            })
    
    all_beta_results.append(beta_summary)
    
    # 即時保存 beta 掃描結果（防止中途中斷）
    scan_results_file = BASE_SAVE_DIR / "beta_scan_results.json"
    with open(scan_results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'scan_info': {
                'beta_range': f"{BETA_VALUES[0]} ~ {BETA_VALUES[-1]}",
                'total_betas': len(BETA_VALUES),
                'completed_betas': beta_idx + 1,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'results': all_beta_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Beta={BETA:.2f} 訓練完成，最佳 UAR: {best_uar * 100:.2f}%")
    print(f"✓ 已更新 beta 掃描結果: {scan_results_file}\n")
    
    # 清理記憶體
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================================
# Beta 掃描完成
# ============================================================================
print("\n" + "="*70)
print("🎉 Beta 掃描完成！")
print("="*70)

# 找出最佳 beta
best_beta_result = max(all_beta_results, key=lambda x: x['best_uar'])

print(f"\n📊 掃描摘要:")
print(f"  測試 Beta 數量: {len(BETA_VALUES)}")
print(f"  範圍: {BETA_VALUES[0]} ~ {BETA_VALUES[-1]}")
print(f"\n🏆 最佳結果:")
print(f"  Beta: {best_beta_result['beta']:.2f}")
print(f"  UAR:  {best_beta_result['best_uar'] * 100:.2f}%")
if 'wa' in best_beta_result:
    print(f"  WA:   {best_beta_result['wa'] * 100:.2f}%")
if 'wer_iemocap' in best_beta_result:
    print(f"  WER (IEMOCAP): {best_beta_result['wer_iemocap'] * 100:.2f}%")
if 'wer_cv' in best_beta_result:
    print(f"  WER (CV): {best_beta_result['wer_cv'] * 100:.2f}%")
print(f"  模型位置: {best_beta_result['model_path']}")

print(f"\n所有 Beta 結果:")
for result in sorted(all_beta_results, key=lambda x: x['beta']):
    uar_pct = result['best_uar'] * 100
    marker = " ⭐" if result == best_beta_result else ""
    print(f"  Beta={result['beta']:.2f}: UAR={uar_pct:.2f}%{marker}")

print(f"\n完整結果已保存至: {BASE_SAVE_DIR / 'beta_scan_results.json'}")
print("="*70)
