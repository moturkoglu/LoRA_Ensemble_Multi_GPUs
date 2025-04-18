from datasets import load_dataset
from transformers import AutoTokenizer

sst2_dataset = load_dataset('glue', 'sst2')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['sentence'], padding="max_length", truncation=True)

sst2_tokenized = sst2_dataset.map(tokenize, batched=True)

sst2_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

sst2_tokenized = sst2_tokenized.rename_column("label", "labels")

train_dataset = sst2_tokenized['train']
test_dataset = sst2_tokenized['test']