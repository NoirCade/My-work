import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('train.csv', encoding='utf-8')
data_train, data_test = train_test_split(df, test_size=0.1, random_state=42)
real_data_test = pd.read_csv('test.csv', encoding='utf-8')
X_train = data_train.text.tolist()
X_test = data_test.text.tolist()
y_train = data_train.target.tolist()
y_test = data_test.target.tolist()

class_names = ['0', '1']
print('size of training set: %s' % (len(data_train['text'])))
print('size of validation set: %s' % (len(data_test['text'])))


# Tokenize the input data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
x_train = tokenizer(X_train, padding=True, truncation=True, max_length=350, return_tensors='pt')
x_test = tokenizer(X_test, padding=True, truncation=True, max_length=350, return_tensors='pt')
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# Create PyTorch Dataset and DataLoader
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(x_train, y_train)
test_dataset = CustomDataset(x_test, y_test)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False)

# Load model
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(class_names))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training and evaluation loop
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
best_acc = 0
for epoch in range(num_epochs):
    model.train()
    for step,(batch) in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}] Step [{step}/{len(train_dataloader)}] Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            _, predictions = torch.max(outputs.logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += labels.size(0)
        accuracy = num_correct / num_samples
        print(f'Epoch {epoch+1} Test Accuracy: {accuracy:.3f}')
        # Save the PyTorch model
        if best_acc < accuracy:
            best_acc = accuracy
            torch.save(model.state_dict(), 'bert_model.pt')

