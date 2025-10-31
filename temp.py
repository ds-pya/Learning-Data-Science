# xlmr_ner_crf.py
from typing import Optional, Dict, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import XLMRobertaConfig
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
from torchcrf import CRF  # pip install pytorch-crf

class XLMRobertaForNERWithCRF(nn.Module):
    """
    XLM-RoBERTa 기반 NER + CRF
    - emissions: (B, T, C)
    - mask: bool (B, T), CRF 제약으로 mask[:,0]은 반드시 True
    - labels: (B, T), 유효 토큰은 [0..C-1], 무시할 곳은 -100 (원본은 보존)
    """
    def __init__(
        self,
        num_labels: int,
        pretrained_name_or_path: str = "xlm-roberta-base",
        dropout_prob: float = 0.1,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        gradient_checkpointing: bool = False,
        pad_label_id: int = 0,          # mask=False 위치에 채울 안전 라벨(O 권장)
        crf_reduction: str = "token_mean"
    ):
        super().__init__()
        self.config = XLMRobertaConfig.from_pretrained(pretrained_name_or_path)
        self.backbone = XLMRobertaModel.from_pretrained(pretrained_name_or_path, config=self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # 라벨 매핑(옵션)
        if id2label is None:
            id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        self.id2label = id2label
        self.label2id = label2id

        # CRF
        self.crf = CRF(num_tags=num_labels, batch_first=True)
        assert crf_reduction in {"none", "sum", "mean", "token_mean"}
        self.crf_reduction = crf_reduction

        self.pad_label_id = pad_label_id  # 보통 O=0

    def _build_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        special_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        attention_mask(1/0), special_token_mask(1=스페셜)를 결합하여 최종 mask 생성.
        CRF 제약상 첫 스텝은 True로 강제.
        """
        if attention_mask is None:
            raise ValueError("attention_mask is required for CRF decoding/training.")
        mask = attention_mask.to(torch.bool).clone()
        if special_token_mask is not None:
            sp = special_token_mask.to(torch.bool).clone()
            # 첫 타임스텝은 CRF 제약 때문에 꺼지면 안 됨
            sp[:, 0] = False
            mask = mask & (~sp)

        # 최종적으로 첫 타임스텝 On (배치 전 시퀀스에서 True여야 함)
        mask[:, 0] = True
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,                       # (B, T)
        attention_mask: Optional[torch.Tensor] = None, # (B, T)
        labels: Optional[torch.Tensor] = None,         # (B, T) [-100 or 0..C-1]
        special_token_mask: Optional[torch.Tensor] = None,  # (B, T) 1=스페셜
        token_type_ids: Optional[torch.Tensor] = None,      # XLM-R은 보통 미사용
        position_ids: Optional[torch.Tensor] = None,        # 필요 시만 전달
        return_probs: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[List[int]]]]:
        """
        Returns:
          - loss (opt): scalar
          - logits: (B, T, C)
          - preds: List[List[int]] (CRF Viterbi 경로)
          - probs (opt): (B, T, C) Softmax (디버그/분석용)
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,      # ← 필요 시만 넘기면 됨
        )
        hidden = self.dropout(outputs.last_hidden_state)  # (B, T, H)
        emissions = self.classifier(hidden)               # (B, T, C)

        mask = self._build_mask(attention_mask, special_token_mask)

        result: Dict[str, Union[torch.Tensor, List[List[int]]]] = {
            "logits": emissions
        }

        if labels is not None:
            # 라벨 원본 보존
            tags = labels.clone().to(torch.long)

            # mask=False 위치는 CRF가 무시하지만, 유효 범위 보장을 위해 안전 라벨로 채움
            tags[~mask] = self.pad_label_id

            # 손실(최대우도 => 음수 부호)
            loss = -self.crf(emissions, tags, mask=mask, reduction=self.crf_reduction)
            result["loss"] = loss

        # 디코딩
        preds = self.crf.decode(emissions, mask=mask)  # List[List[int]]
        result["preds"] = preds

        if return_probs:
            # 참고: CRF의 경로확률과는 다릅니다. 여기서는 emissions의 토큰별 softmax만 제공
            result["probs"] = F.softmax(emissions, dim=-1)

        return result