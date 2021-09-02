from sklearn.metrics import classification_report, f1_score, accuracy_score
from model import EntityModel
import config

if __name__ == "__main__":
    df_data = get_data(config.TRAINING_FILE)

    sentences, tag, enc_tag = process_data(df_data)

    (train_sentences, test_sentences, train_tag, test_tag) = model_selection.train_test_split(
        sentences, tag, random_state=42, test_size=0.3)

    valid_dataset = EntityDataset(
        texts=test_sentences, tags=test_tag)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batch_size, num_workers=1)

    device = torch.device("cuda")
    model = EntityModel(enc_tag=enc_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    predictions, true_labels = valid(model, valid_data_loader, device, enc_tag)
    print("Accuracy score: %f" % (accuracy_score(np.concatenate(true_labels), np.concatenate(predictions))))
    print(classification_report(np.concatenate(true_labels), np.concatenate(predictions)))
