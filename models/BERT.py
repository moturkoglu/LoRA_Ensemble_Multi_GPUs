#!/usr/bin/env python

"""
Implements LoRA for the BERT model
"""

from typing import List

from transformers import AutoModelForSequenceClassification
from torch import nn

from models.lora import Init_Weight, EnsembleLoRA

class BertEnsemble(nn.Module):
    def __init__(self, n_members, model_name):
        super(BertEnsemble, self).__init__()

        self.n_members = n_members
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print(model.bert.encoder.layer)
        self.models = nn.ModuleList([AutoModelForSequenceClassification.from_pretrained(model_name) for _ in range(n_members)])

        pass

    def forward(self, inputs):
        out = [model(**inputs) for model in self.models]

        return out

class LoRABert(nn.Module):
    def __init__(self,
                 bert_model,
                 rank: int,
                 n_members: int,
                 lora_layers: List[int] = None,
                 lora_init: Init_Weight = Init_Weight.DEFAULT,
                 init_settings: dict = None,
                 chunk_size: int = None
                 ) -> None:
        super(LoRABert, self).__init__()

        self.bert_model = bert_model

        self.n_members = n_members

        self.chunk_size = chunk_size

        if lora_layers is None:
            self.lora_layers = list(range(len(self.bert_model.model.bert.encoder.layer)))
        else:
            self.lora_layers = lora_layers

        for param in self.bert_model.parameters():
            param.requires_grad = False

        for layer_id, enc_layer in enumerate(self.bert_model.model.bert.encoder.layer):
            if layer_id not in self.lora_layers:
                continue

            enc_layer.attention.self.query = EnsembleLoRA(
                w=enc_layer.attention.self.query,
                rank=rank,
                dim=enc_layer.attention.self.query.weight.shape[0],
                n_members=n_members,
                initialize=True,
                init_type=lora_init,
                init_settings=init_settings,
                chunk_size=chunk_size
            )
            enc_layer.attention.self.key = EnsembleLoRA(
                w=enc_layer.attention.self.key,
                rank=rank,
                dim=enc_layer.attention.self.key.weight.shape[0],
                n_members=n_members,
                initialize=True,
                init_type=lora_init,
                init_settings=init_settings,
                chunk_size=chunk_size
            )
            enc_layer.attention.self.value = EnsembleLoRA(
                w=enc_layer.attention.self.value,
                rank=rank,
                dim=enc_layer.attention.self.value.weight.shape[0],
                n_members=n_members,
                initialize=True,
                init_type=lora_init,
                init_settings=init_settings,
                chunk_size=chunk_size
            )

            enc_layer.attention.output.dense = EnsembleLoRA(
                w=enc_layer.attention.output.dense,
                rank=rank,
                dim=enc_layer.attention.output.dense.weight.shape[0],
                n_members=n_members,
                initialize=True,
                init_type=lora_init,
                init_settings=init_settings,
                chunk_size=chunk_size
            )

    def forward(self, inputs):

        inputs = {key: values.repeat_interleave(self.n_members, dim=0) for key, values in inputs.items()}

        out = self.bert_model(inputs)

        out = out.view(out.shape[0] // self.n_members, self.n_members, -1)
        out = out.permute(1, 0, 2)

        return out

class BertModel(nn.Module):
    def __init__(self, model_name):
        super(BertModel, self).__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, inputs):
        out = self.model(**inputs)

        return out
