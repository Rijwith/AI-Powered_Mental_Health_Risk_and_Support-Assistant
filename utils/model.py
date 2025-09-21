# utils/model.py
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, RobertaPreTrainedModel

class DistressModel(RobertaPreTrainedModel):
    def __init__(self, num_labels=5, pretrained_model='roberta-base', pos_weight=None):
        config = RobertaConfig.from_pretrained(pretrained_model, num_labels=num_labels)
        super().__init__(config)
        
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.pos_weight = pos_weight

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(logits, labels.float())
        
        return {'loss': loss, 'logits': logits}
