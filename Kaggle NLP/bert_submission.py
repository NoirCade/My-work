import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
counter = 1

# Load the saved model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
model.load_state_dict(torch.load('D:/models/0307 1e-5, 32batch/bert_model_epoch25.pth'))

# Load the CSV file of input texts
data_test = pd.read_csv('test.csv', encoding='utf-8')

# Tokenize the input texts and convert to PyTorch tensors
X_test = [tokenizer.encode(text, add_special_tokens=True, max_length=350, truncation=True, padding='max_length')
          for text in data_test['text'].tolist()]
X_test = torch.tensor(X_test, dtype=torch.long).to(device)

# Pass the tokenized text through the model to get the predictions
model.eval()
with torch.no_grad():
    y_pred = []
    for batch in tqdm(X_test.split(32), desc="Predicting"):
        outputs = model(batch)
        y_pred += outputs[0].argmax(axis=1).tolist()

# Write the predictions to a new CSV file
predictions = pd.DataFrame({'id': data_test['id'].tolist(), 'target': y_pred})
predictions.to_csv(f'predictions{counter}.csv', index=False)
