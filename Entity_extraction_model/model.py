import config
import torch
import transformers
import torch.nn as nn


class EntityModel(nn.Module):

    def __init__(self, enc_tag):
        super(EntityModel, self).__init__()

        self.num_tag = len(enc_tag.classes_)
        self.config = transformers.AutoConfig.from_pretrained(config.BASE_MODEL_PATH)
        self.config._num_labels = self.num_tag
        self.config.label2id = {k: v for k, v in zip(enc_tag.classes_, enc_tag.transform(enc_tag.classes_))}
        self.config.id2label = {k: v for k, v in zip(enc_tag.transform(enc_tag.classes_), enc_tag.classes_)}
        self.classifier = transformers.AutoModelForTokenClassification.from_config(self.config)

    def forward(self, ids, mask, token_type_ids, target_tags):
        output_1 = self.classifier(ids, attention_mask=mask, token_type_ids=token_type_ids, labels=target_tags)
        return output_1
