import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len):
        with open(json_file, 'r') as f:
            self.data = json.load(f)  # Load the entire JSON file as a list of objects
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        input_text = f"{item['instruction']} {item['input']}"
        labels_text = item['output']

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        labels = self.tokenizer.encode_plus(
            labels_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels['input_ids'], dtype=torch.long)
        }

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the dataset
train_dataset = CustomDataset('/home/alucard/venv/Psychology-10K.json', tokenizer, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Load the pre-trained model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained('./finetuned_t5')
tokenizer.save_pretrained('./finetuned_t5')
