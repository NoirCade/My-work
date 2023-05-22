from transformers import BertForSequenceClassification
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from customdataset import MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


dataset = MyDataset('./train.csv')
train_dataset, val_dataset = dataset.train_val_split()
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Define the training loop
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels =inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(dataloader)

# Train the model for a certain number of epochs
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    print('Epoch {}/{} - Train loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct/len(dataloader.dataset)
    besttest = accuracy
    best_accuracy = 0
    if besttest >= best_accuracy:
        best_accuracy = besttest
        torch.save(model.state_dict(), 'bestmodel.pth')

    return total_loss/len(dataloader), accuracy



val_loss, val_acc = evaluate(model, val_dataloader, criterion)
print('Val loss: {:.4f} - Val acc: {:.4f}'.format(val_loss, val_acc))