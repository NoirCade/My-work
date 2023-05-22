import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the pre-trained BERT model with a classification head
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Load the data
df = pd.read_csv('data.csv')

# Tokenize the text data
tokenized_texts = [tokenizer.encode(text, add_special_tokens=True) for text in df['text']]

# Pad and truncate the tokenized sequences to a fixed length of 128
max_len = 128
input_ids = torch.tensor(
    [tokenizer.convert_tokens_to_ids(text[:max_len]) + [0] * (max_len - len(text)) for text in tokenized_texts])
attention_masks = torch.tensor([[int(token_id > 0) for token_id in text] for text in input_ids])

# Split the data into train and validation sets
labels = torch.tensor(df['label'].tolist())
train_inputs, val_inputs, train_labels, val_labels, train_masks, val_masks = train_test_split(input_ids, labels,
                                                                                              attention_masks,
                                                                                              test_size=0.2)

# Create PyTorch DataLoader objects
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=32)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(3):
    train_loss = 0.0
    val_loss = 0.0

    # Training loop
    model.train()
    for inputs, masks, labels in train_dataloader:
        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs[0]
        train_loss += loss.item() * len(inputs)

        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    with torch.no_grad():
        for inputs, masks, labels in val_dataloader:
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)

            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs[0]
            val_loss += loss.item() * len(inputs)

    # Calculate average training and validation loss for the epoch
    train_loss /= len(train_inputs)
    val_loss /= len(val_inputs)
    print(f'Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')