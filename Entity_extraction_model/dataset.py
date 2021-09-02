import config
import torch


class EntityDataset:

    def __init__(self, texts, tags, enc_tag):
        self.texts = texts
        self.tags = tags
        self.enc_tag = enc_tag

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item, max_len=512):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tags = []

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )

            input_len = len(inputs)
            ids.extend(inputs)
            target_tags.extend([tags[i]] * input_len)

        ids = ids[: max_len - 2]
        target_tags = target_tags[: max_len - 2]

        ids = [101] + ids + [102]
        target_tags = [self.enc_tag.transform(['O'])[0]] + target_tags + [self.enc_tag.transform(['O'])[0]]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = max_len - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tags = target_tags + ([self.enc_tag.transform(['O'])[0]] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tags": torch.tensor(target_tags, dtype=torch.long)
        }
