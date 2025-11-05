import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Tuple, Union

class WhisperForEmotionRecognition(WhisperForConditionalGeneration):
    """
    擴展 Whisper 以支援句子級和詞級情緒識別
    實現論文中的 SAC (Self-Attention Causal) mask
    
    🔧 修正版：正確實現 SAC mask
    - WLE 不能看到 SLE（防止直接複製）
    - SLE 可以看到所有之前的 token（包括音訊特徵）
    - 保持因果性（不能看到未來）
    """

    def __init__(self, config: WhisperConfig, sle_token_ids: dict, wle_token_ids: dict):
        super().__init__(config)
        
        self.sle_tokens = sle_token_ids
        self.wle_tokens = wle_token_ids
        self.all_sle_ids_set = set(self.sle_tokens.values()) - {None}
        self.all_wle_ids_set = set(self.wle_tokens.values()) - {None}
        
        self.use_sac_mask_in_training = True
        self.use_sac_mask_in_inference = False
        
        self._eos_token_id = None
        self._pad_token_id = None
        
        print(f"✓ 模型初始化完成 (dtype: {self.dtype})")
        print(f"  - SLE tokens: {len(self.all_sle_ids_set)} 個")
        print(f"  - WLE tokens: {len(self.all_wle_ids_set)} 個")

    def create_sac_mask_vectorized(
        self, 
        decoder_input_ids: torch.Tensor, 
        causal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        建立 SAC mask（修正版）
        
        關鍵邏輯：
        1. 只遮罩 WLE query -> SLE key 的連接
        2. SLE 本身可以看到所有之前的 token
        3. 不影響 cross-attention（SLE 必須能看到音訊）
        
        Args:
            decoder_input_ids: [batch, seq_len]
            causal_mask: [batch, 1, seq_len, seq_len] 基礎因果遮罩
        
        Returns:
            combined_mask: [batch, 1, seq_len, seq_len]
        """
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # 如果沒有情緒 token，直接返回因果遮罩
        if len(self.all_sle_ids_set) == 0 or len(self.all_wle_ids_set) == 0:
            return causal_mask
        
        # 建立情緒 token 的位置標記
        all_sle_ids = torch.tensor(
            list(self.all_sle_ids_set), 
            device=device, 
            dtype=torch.int32
        )
        all_wle_ids = torch.tensor(
            list(self.all_wle_ids_set), 
            device=device, 
            dtype=torch.int32
        )
        
        decoder_input_ids_int32 = decoder_input_ids.to(torch.int32)
        
        # 標記哪些位置是 WLE (query) 和 SLE (key)
        is_wle_query = torch.isin(decoder_input_ids_int32, all_wle_ids)  # [batch, seq_len]
        is_sle_key = torch.isin(decoder_input_ids_int32, all_sle_ids)    # [batch, seq_len]
        
        # 建立禁止矩陣：WLE query 不能看到 SLE key
        # [batch, seq_len, 1] & [batch, 1, seq_len] -> [batch, seq_len, seq_len]
        wle_mask = is_wle_query.unsqueeze(2)  # [batch, seq_len, 1]
        sle_mask = is_sle_key.unsqueeze(1)     # [batch, 1, seq_len]
        forbidden_mask = wle_mask & sle_mask   # [batch, seq_len, seq_len]
        
        # 🔧 關鍵：只在因果遮罩允許的範圍內應用 SAC
        # 因為 WLE 只出現在對應詞的後面，SLE 在序列開頭
        # 所以這個遮罩只會影響 WLE 看 SLE 的情況
        
        # 建立 SAC 遮罩值
        sac_mask = torch.zeros(
            (batch_size, 1, seq_len, seq_len), 
            dtype=causal_mask.dtype, 
            device=device
        )
        mask_value = torch.finfo(causal_mask.dtype).min
        
        # 在禁止的位置填入 mask_value
        sac_mask = sac_mask.masked_fill(forbidden_mask.unsqueeze(1), mask_value)
        
        # 確保 causal_mask 的 batch 維度正確
        if causal_mask.shape[0] == 1:
            causal_mask = causal_mask.expand(batch_size, -1, -1, -1)
        
        # 組合因果遮罩和 SAC 遮罩
        # 使用 minimum 確保兩個遮罩都生效
        combined_mask = torch.minimum(causal_mask, sac_mask)
        
        return combined_mask

    def _should_use_sac_mask(self) -> bool:
        """根據訓練/推理模式決定是否使用 SAC mask"""
        return self.use_sac_mask_in_training if self.training else self.use_sac_mask_in_inference
    
    def _get_past_key_values_length(self, past_key_values) -> int:
        """獲取 KV cache 的長度"""
        if past_key_values is None:
            return 0
        if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) == 0:
            return 0
        first_layer_cache = past_key_values[0]
        if first_layer_cache is None:
            return 0
        if not isinstance(first_layer_cache, (tuple, list)) or len(first_layer_cache) == 0:
            return 0
        key_tensor = first_layer_cache[0]
        if key_tensor is None:
            return 0
        return key_tensor.shape[2]

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        force_sac_mask: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        前向傳播
        
        🔧 關鍵點：
        1. decoder_attention_mask 只影響 self-attention
        2. cross-attention 不受影響（SLE 必須能看到音訊）
        3. SAC mask 在 decoder self-attention 中生效
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 決定是否使用 SAC mask
        use_sac_mask = force_sac_mask if force_sac_mask is not None else self._should_use_sac_mask()
        
        # ============================================================
        # 步驟 1: Encoder - 處理音訊特徵
        # ============================================================
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        encoder_hidden_states = encoder_outputs[0]
        
        # ============================================================
        # 步驟 2: 準備 Decoder 輸入
        # ============================================================
        if decoder_input_ids is not None and decoder_inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds")
        
        past_key_values_length = self._get_past_key_values_length(past_key_values)
        
        if decoder_input_ids is not None:
            batch_size, seq_len = decoder_input_ids.shape
        elif decoder_inputs_embeds is not None:
            batch_size, seq_len = decoder_inputs_embeds.shape[:2]
        else:
            batch_size = input_features.shape[0]
            seq_len = 1

        # 獲取 decoder embeddings
        if decoder_inputs_embeds is None:
            if decoder_input_ids is None:
                decoder_input_ids = torch.tensor(
                    [[self.config.decoder_start_token_id]], 
                    device=input_features.device, 
                    dtype=torch.long
                ).expand(batch_size, -1)
            
            decoder_inputs_embeds = self.model.decoder.embed_tokens(decoder_input_ids)
            decoder_inputs_embeds = decoder_inputs_embeds * self.model.decoder.embed_scale

        # ============================================================
        # 步驟 3: 建立 Self-Attention Mask
        # ============================================================
        full_seq_len = seq_len + past_key_values_length
        
        # 3.1 基礎因果遮罩（下三角）
        causal_mask = torch.zeros(
            (batch_size, 1, seq_len, full_seq_len), 
            dtype=decoder_inputs_embeds.dtype, 
            device=decoder_inputs_embeds.device
        )
        
        # 建立上三角遮罩（禁止看到未來）
        mask = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.bool, device=decoder_inputs_embeds.device), 
            diagonal=1
        )
        mask_value = torch.finfo(decoder_inputs_embeds.dtype).min
        causal_mask[:, :, :, past_key_values_length:].masked_fill_(mask, mask_value)

        # 3.2 加入 padding mask（如果有）
        if decoder_attention_mask is not None:
            expanded_padding_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(2)
            expanded_padding_mask = expanded_padding_mask.to(dtype=decoder_inputs_embeds.dtype)
            expanded_padding_mask = (1.0 - expanded_padding_mask) * mask_value
            causal_mask = causal_mask + expanded_padding_mask
        
        # 3.3 加入 SAC mask（如果啟用且在訓練時）
        # 🔧 關鍵：只在沒有 KV cache 時應用（完整序列訓練）
        if use_sac_mask and decoder_input_ids is not None and past_key_values_length == 0:
            final_decoder_mask = self.create_sac_mask_vectorized(
                decoder_input_ids, 
                causal_mask
            )
            
            # Debug 輸出（可選）
            if self.training and torch.rand(1).item() < 0.01:  # 1% 機率印出
                print(f"\n[SAC Mask Debug]")
                print(f"  - decoder_input_ids shape: {decoder_input_ids.shape}")
                print(f"  - SAC mask applied: True")
                print(f"  - SLE positions: {torch.isin(decoder_input_ids, torch.tensor(list(self.all_sle_ids_set), device=decoder_input_ids.device)).sum().item()}")
                print(f"  - WLE positions: {torch.isin(decoder_input_ids, torch.tensor(list(self.all_wle_ids_set), device=decoder_input_ids.device)).sum().item()}")
        else:
            final_decoder_mask = causal_mask
        
        final_decoder_mask = final_decoder_mask.contiguous()
        
        # ============================================================
        # 步驟 4: Decoder 前向傳播
        # 🔧 注意：final_decoder_mask 只影響 self-attention
        #          cross-attention 使用 encoder_hidden_states 不受影響
        # ============================================================
        decoder_kwargs = {
            "input_ids": None,
            "attention_mask": final_decoder_mask,  # ← Self-attention mask
            "encoder_hidden_states": encoder_hidden_states,  # ← Cross-attention 輸入
            "head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": decoder_inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        
        if cache_position is not None:
            decoder_kwargs["cache_position"] = cache_position
        
        decoder_outputs = self.model.decoder(**decoder_kwargs)
        
        # ============================================================
        # 步驟 5: 投影到詞彙表
        # ============================================================
        lm_logits = self.proj_out(decoder_outputs[0])
        
        # ============================================================
        # 步驟 6: 計算 Loss（如果有 labels）
        # ============================================================
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # 確保形狀匹配
            min_len = min(lm_logits.shape[1], labels.shape[1])
            lm_logits_for_loss = lm_logits[:, :min_len, :]
            labels_for_loss = labels[:, :min_len]
            
            loss = loss_fct(
                lm_logits_for_loss.reshape(-1, self.config.vocab_size),
                labels_for_loss.reshape(-1)
            )
        
        # ============================================================
        # 步驟 7: 返回結果
        # ============================================================
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    def generate(
        self,
        input_features,
        decoder_input_ids=None,
        max_length=None,
        suppress_tokens=None,
        repetition_penalty=3.0,  # 默認值調整
        no_repeat_ngram_size=3,  # 默認值調整
        use_manual_generation=True,
        eos_token_id=None,  # === 新增 ===
        pad_token_id=None,  # === 新增 ===
        **kwargs
    ):
        """
        擴展的 generate 方法
        
        Args:
            eos_token_id: EOS token ID（如果不提供，會嘗試從 config 獲取）
            pad_token_id: PAD token ID（如果不提供，會使用 eos_token_id）
        """
        # === 設置 token IDs ===
        if eos_token_id is not None:
            self._eos_token_id = eos_token_id
        elif self._eos_token_id is None:
            # 嘗試從 config 獲取
            self._eos_token_id = getattr(self.config, 'eos_token_id', 50257)
        
        if pad_token_id is not None:
            self._pad_token_id = pad_token_id
        elif self._pad_token_id is None:
            # 默認使用 eos_token_id 作為 pad
            self._pad_token_id = self._eos_token_id
        
        was_training = self.training
        if was_training:
            self.eval()
        
        original_attn_impl = getattr(self.config, '_attn_implementation', 'eager')
        self.config._attn_implementation = "eager"
        
        try:
            if use_manual_generation:
                outputs = self._manual_generate(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    max_length=max_length or 100,  # 默認 100
                    suppress_tokens=suppress_tokens,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            else:
                kwargs['use_cache'] = False
                outputs = super().generate(
                    input_features=input_features,
                    decoder_input_ids=decoder_input_ids,
                    max_length=max_length,
                    suppress_tokens=suppress_tokens,
                    **kwargs
                )
            return outputs
        finally:
            self.config._attn_implementation = original_attn_impl
            if was_training:
                self.train()
    
    def _manual_generate(
        self,
        input_features,
        decoder_input_ids,
        max_length=100,
        suppress_tokens=None,
        repetition_penalty=3.0,
        no_repeat_ngram_size=3,
    ):
        """手動逐步生成"""
        batch_size = input_features.shape[0]
        device = input_features.device
        
        current_ids = decoder_input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 準備 suppress mask
        suppress_mask = None
        if suppress_tokens:
            suppress_mask = torch.zeros(self.config.vocab_size, device=device)
            for token_id in suppress_tokens:
                if token_id is not None:
                    suppress_mask[token_id] = float('-inf')
        
        for step in range(max_length):
            if finished.all():
                break
            
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.forward(
                    input_features=input_features,
                    decoder_input_ids=current_ids,
                    labels=None,
                    use_cache=False
                )
            
            next_token_logits = outputs.logits[:, -1, :].clone()
            
            if suppress_mask is not None:
                next_token_logits = next_token_logits + suppress_mask
            
            if repetition_penalty > 1.0 and step > 3:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, current_ids, finished, repetition_penalty
                )
            
            if no_repeat_ngram_size > 0 and step > no_repeat_ngram_size:
                next_token_logits = self._apply_no_repeat_ngram(
                    next_token_logits, current_ids, finished, no_repeat_ngram_size
                )
            
            # 長度懲罰
            if step > max_length * 0.5:
                boost = (step - max_length * 0.5) * 0.5
                next_token_logits[:, self._eos_token_id] += boost
            
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            is_eos = (next_token_ids.squeeze(-1) == self._eos_token_id)
            finished = finished | is_eos
            
            next_token_ids[finished] = self._pad_token_id
            current_ids = torch.cat([current_ids, next_token_ids], dim=1)
        
        return current_ids
    
    def _apply_repetition_penalty(self, logits, current_ids, finished, penalty_strength):
        """應用重複懲罰"""
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            if not finished[i]:
                window_size = min(20, current_ids.shape[1] - 4)
                if window_size > 0:
                    recent_tokens = current_ids[i, -window_size:].tolist()
                    token_counts = {}
                    for token in recent_tokens:
                        token_counts[token] = token_counts.get(token, 0) + 1
                    
                    for token, count in token_counts.items():
                        if count >= 2:
                            penalty = penalty_strength * (count ** 2)
                            logits[i, token] -= penalty
        
        return logits
    
    def _apply_no_repeat_ngram(self, logits, current_ids, finished, ngram_size):
        """禁止重複 n-gram"""
        batch_size = logits.shape[0]
        
        for i in range(batch_size):
            if not finished[i]:
                seq_len = current_ids.shape[1]
                if seq_len >= ngram_size:
                    prefix = tuple(current_ids[i, -(ngram_size-1):].tolist())
                    
                    for j in range(max(0, seq_len - ngram_size - 10), seq_len - ngram_size + 1):
                        if tuple(current_ids[i, j:j+ngram_size-1].tolist()) == prefix:
                            banned_token = current_ids[i, j+ngram_size-1].item()
                            logits[i, banned_token] = float('-inf')
        
        return logits

    def gradient_checkpointing_enable(self):
        super().gradient_checkpointing_enable()
        print("✓ Gradient checkpointing 已啟用")
    
    def set_sac_mask_config(self, use_in_training: bool = True, use_in_inference: bool = False):
        self.use_sac_mask_in_training = use_in_training
        self.use_sac_mask_in_inference = use_in_inference
        print(f"✓ SAC mask 配置已更新:")
        print(f"  - 訓練時: {use_in_training}")
        print(f"  - 推理時: {use_in_inference}")