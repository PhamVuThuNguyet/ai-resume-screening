import json
import re
import config
from dataset import EntityDataset
from model import EntityModel
import torch
import numpy as np
import joblib


# JSON formatting functions
def convert_data_to_spacy(JSON_FilePath):
    training_data = []
    with open(JSON_FilePath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = json.loads(line)
        text = data['data']
        data_annotations = data['label']
        training_data.append((text, {"entities": data_annotations}))
    return training_data


def trim_entity_spans(data: list) -> list:
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """
    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data


def clean_entities(training_data):
    clean_data = []
    for text, annotation in training_data:

        entities = annotation.get('entities')
        entities_copy = entities.copy()

        # append entity only if it is longer than its overlapping entity
        i = 0
        for entity in entities_copy:
            j = 0
            for overlapping_entity in entities_copy:
                # Skip self
                if i != j:
                    e_start, e_end, oe_start, oe_end = entity[0], entity[1], \
                                                       overlapping_entity[0], overlapping_entity[1]
                    # Delete any entity that overlaps, keep if longer
                    if ((oe_start <= e_start <= oe_end) or (oe_end >= e_end >= oe_start)) \
                            and ((e_end - e_start) <= (oe_end - oe_start)):
                        entities.remove(entity)
                j += 1
            i += 1
        clean_data.append((text, {'entities': entities}))

    return clean_data


def process_data(df):

    enc_tag = preprocessing.LabelEncoder()
    all_ents = df['entities_mapped'].apply(pd.Series).stack().values
    enc_tag.fit(all_ents)
    sentences = list(df["clean_content"].str.split())
    tag = list(df["entities_mapped"].apply(enc_tag.transform))

    return sentences, tag, enc_tag


def get_data(df_path):

    data = trim_entity_spans(convert_data_to_spacy(df_path))
    data = clean_entities(data)
    df_data = pd.DataFrame(columns=["clean_content", "entities_mapped"])
    entities_mapped = []
    clean_content = []
    for i in range(len(data)):
        content = data[i][0].split()
        entities = data[i][1]["entities"]
        words = []
        labels = []

        for word in content:
            words.append(word)
            found = False
            for entity in sorted(entities):
                ent_start = entity[0]
                ent_end = entity[1]
                ent_label = entity[2]
                if word in data[i][0][ent_start: ent_end].split():
                    labels.append(ent_label)
                    found = True
                    break
            if not found:
                labels.append("O")

        entities_mapped.append(labels)
        clean_content.append(words)

    df_data["entities_mapped"] = entities_mapped
    df_data["clean_content"] = clean_content
    df_data["clean_content"] = df_data["clean_content"].apply(lambda x: " ".join(x))
    return df_data


def valid(model, testing_loader, device, enc_tag):
    model.eval()
    predictions_, true_labels_ = [], []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            input_mask = data['mask']
            targets = data['target_tags']
            for k, v in data.items():
                data[k] = v.to(device, dtype=torch.long)
            output = model(**data)
            loss, logits = output[:2]
            logits = logits.detach().cpu().numpy()

            logits = [list(p) for p in np.argmax(logits, axis=2)]
            input_mask = input_mask.to('cpu').numpy()
            label_ids = targets.to('cpu').numpy()
            for i, mask in enumerate(input_mask):
                temp_1 = []  # Real one
                temp_2 = []  # Predict one

                for j, m in enumerate(mask):
                    # Mark=0, meaning its a pad word, don't compare
                    if m:
                        temp_1.append(enc_tag.inverse_transform([label_ids[i][j]])[0])
                        temp_2.append(enc_tag.inverse_transform([logits[i][j]])[0])
                    else:
                        break
                true_labels_.append(temp_1)
                predictions_.append(temp_2)

    return predictions_, true_labels_


def predict(sentence):
    
    meta_data = joblib.load("./meta.bin")
    enc_tag = meta_data["enc_tag"]
    tokenized_sentence = config.TOKENIZER.encode(sentence)
    sentence = sentence.split()
    test_dataset = EntityDataset(
        texts=[sentence],
        tags=[[1] * len(sentence)],
        enc_tag=enc_tag
    )
    valid_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=1)
    device = torch.device("cuda")
    model = EntityModel(enc_tag=enc_tag)
    model.load_state_dict(torch.load('./bert-base-uncased.bin'))
    model.to(device)
    predictions, true_labels = valid(model, valid_data_loader, device, enc_tag)
    print(sentence)
    print(tokenized_sentence)
    print(predictions)
    return {
        'sentence': sentence,
        'tokenized_sentence': tokenized_sentence,
        'predictions': predictions
    }
