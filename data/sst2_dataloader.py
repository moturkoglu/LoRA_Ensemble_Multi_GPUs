from datasets import load_dataset, DownloadMode
from transformers import AutoTokenizer, BertTokenizer

sst2_dataset = load_dataset('glue', 'sst2', download_mode="reuse_dataset_if_exists")

# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['sentence'], padding="max_length", truncation=True, max_length=128)

sst2_tokenized = sst2_dataset.map(tokenize, batched=True)

sst2_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

sst2_tokenized = sst2_tokenized.rename_column("label", "labels")

train_dataset = sst2_tokenized['train']
test_dataset = sst2_tokenized['validation']
#test_dataset = sst2_tokenized['test']
#print("Train dataset size:", len(train_dataset))
#print("Validation dataset size:", len(validation_dataset))
#print("Test dataset size:", len(test_dataset))


# Print unique labels in train and test datasets
#train_labels = set(train_dataset['labels'].tolist())
#validation_labels = set(validation_dataset['labels'].tolist())
#test_labels = set(test_dataset['labels'].tolist())
#
#print("Unique labels in train dataset:", train_labels)
#print("Unique labels in validation dataset:", validation_labels)
#print("Unique labels in test dataset:", test_labels)