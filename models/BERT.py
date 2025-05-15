from typing import List, Dict
import enum

import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertSelfAttention

from models.lora import Init_Weight, EnsembleLoRA, BERTEnsembleLoRA, SimpleBERTEnsembleLoRA
from models.vision_transformer import EnsembleHead, Init_Head


class BatchMode(enum.Enum):
    REPEAT = 0
    SPLIT = 1
    DEFAULT = REPEAT


class BertModel(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    def forward(self, inputs):
        return self.model(**inputs)


class LoRAAttention(nn.Module):
    def __init__(self, self_attn: BertSelfAttention, n_members: int):
        super().__init__()
        self.self_attn  = self_attn
        self.n_members = n_members

    def forward(self, *args, **kwargs):
        # args[0] is hidden_states, args[1] might be attention_mask
        # also may be passed as kwargs["attention_mask"]
        if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
            kwargs["attention_mask"] = kwargs["attention_mask"].repeat_interleave(self.n_members, dim=0)
        elif len(args) >= 2 and args[1] is not None:
            hs, attn_mask, *rest = args
            attn_mask = attn_mask.repeat_interleave(self.n_members, dim=0)
            args = (hs, attn_mask, *rest)

        return self.self_attn(*args, **kwargs)



class LoRABert(nn.Module):
    """
    LoRA Ensemble wrapper for BERT for sequence classification.
    """
    def __init__(
        self,
        bert_model: AutoModelForSequenceClassification,
        rank: int,
        n_members: int,
        lora_layers: List[int] = None,
        lora_init: Init_Weight = Init_Weight.DEFAULT,
        init_settings: dict = None,
        chunk_size: int = None,
        init_head: Init_Head = Init_Head.DEFAULT,
        head_settings: dict = None,
        batch_mode: BatchMode = BatchMode.DEFAULT,
    ):
        super().__init__()
        self.bert_model = bert_model
        self.n_members  = n_members
        self.chunk_size = chunk_size
        self.batch_mode = batch_mode

        # ---- config & freeze ----
        cfg         = bert_model.model.config
        hidden_size = cfg.hidden_size
        num_labels  = cfg.num_labels

        encoder = bert_model.model.bert
        for p in encoder.parameters():
            p.requires_grad = False

        # ---- replace classification head ----
        ensemble_head = EnsembleHead(
            hidden_size, num_labels, n_members, init_head, head_settings
        )
        bert_model.model.classifier = ensemble_head

        # ---- figure out which layers to adapt ----
        total = cfg.num_hidden_layers
        self.lora_layers = list(range(total)) if lora_layers is None else lora_layers
        layers = encoder.encoder.layer if hasattr(encoder, "encoder") else encoder.layer

        # ---- inject SimpleBERTEnsembleLoRA adapters ----
        for idx, layer in enumerate(layers):
            if idx not in self.lora_layers:
                continue

            sa: BertSelfAttention = layer.attention.self

            # query/key/value
            for proj in ("query", "key", "value"):
                orig = getattr(sa, proj)
                setattr(
                    sa, proj,
                    SimpleBERTEnsembleLoRA(
                        w=orig,
                        rank=rank,
                        dim=orig.weight.shape[0],
                        n_members=n_members,
                        initialize=True,
                        init_type=lora_init,
                        init_settings=init_settings,
                        out_dim=hidden_size
                    )
                )

            # output projection
            outp = layer.attention.output.dense
            layer.attention.output.dense = SimpleBERTEnsembleLoRA(
                w=outp,
                rank=rank,
                dim=outp.weight.shape[0],
                n_members=n_members,
                initialize=True,
                init_type=lora_init,
                init_settings=init_settings,
                out_dim=hidden_size
            )

        # ---- report trainable params ----
        tp = [(n, p.shape) for n,p in self.named_parameters() if p.requires_grad]
        total_p = sum(p.numel() for _,p in self.named_parameters() if p.requires_grad)
        print(f"[LoRABert] Trainable parameters ({len(tp)} tensors, {total_p} values):")
        for name, shape in tp:
            print(f"  - {name}: {shape}")


    def forward(self, *args, **inputs) -> Tensor:
        # allow dict positional or kwargs
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                inputs = args[0]
            else:
                raise ValueError("LoRABert.forward: expected single dict arg or kwargs")

        # keep your pop
        inputs.pop("labels", None)

        # optional repeat
        if self.batch_mode == BatchMode.REPEAT:
            inputs = {
                k: v.repeat_interleave(self.n_members, dim=0)
                for k,v in inputs.items()
            }

        outputs = self.bert_model.model(**inputs)
        logits  = outputs.logits  # [batch*n_members, num_labels]


        # reshape → [n_members, batch, num_labels]
        batch  = logits.size(0) // self.n_members
        return logits.view(batch, self.n_members, -1).permute(1, 0, 2)


    def gather_params(self) -> Dict[str, Tensor]:
        params: Dict[str, Tensor] = {}
        for name, p in self.named_parameters():
            # include all LoRA adapters’ weights
            if ".w_a." in name or ".w_b." in name:
                params[name] = p
            # include ensemble‐head weights (where you defined them)
            elif "classifier.heads" in name:
                params[name] = p
        return params


    def set_params(self, state_dict: Dict[str, Tensor]) -> None:
        own = {n:p for n,p in self.named_parameters()}
        for name, tensor in state_dict.items():
            if name in own:
                own[name].data.copy_(tensor)