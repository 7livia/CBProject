import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
dataset_path = 'C:/Users/Livia/Desktop/augmented_dataset.csv'
data = pd.read_csv(dataset_path)

# Check the dataset columns and size
print(data.head())
print(f"Number of samples in the dataset: {len(data)}")

# Define a custom Dataset class for handling the data
class ChatbotDataset(Dataset):
    def __init__(self, data):
        self.inputs = data['Pyetje']
        self.outputs = data['Pergjigje']
        self.label_map = {label: idx for idx, label in enumerate(set(self.outputs))}
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_text = self.outputs[idx]
        inputs = self.tokenizer(input_text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
        input_ids = inputs['input_ids'].squeeze()
        label = self.label_map[output_text]
        return {'input_ids': input_ids, 'labels': torch.tensor(label)}

# Create the dataset and dataloader
train_dataset = ChatbotDataset(data)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Check the number of batches
print(f"Number of batches in DataLoader: {len(train_loader)}")

# Print out a few batches to inspect
for i, batch in enumerate(train_loader):
    if i < 2:  # Print the first 2 batches
        print(f"Batch {i + 1} size: {len(batch['input_ids'])}")
    else:
        break

# Load pre-trained mBERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=len(train_dataset.label_map))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Function to calculate accuracy
def calculate_accuracy(predictions, labels):
    preds = torch.argmax(predictions, dim=1)
    return torch.sum(preds == labels).item() / len(labels)

# Lists to store loss and accuracy for each epoch
loss_history = []
accuracy_history = []

# Training loop
num_epochs = 15
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        accuracy = calculate_accuracy(logits, labels)
        epoch_accuracy += accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    avg_accuracy = epoch_accuracy / len(train_loader)
    
    loss_history.append(avg_loss)
    accuracy_history.append(avg_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

# Save the fine-tuned model
model.save_pretrained('./fine_tuned_bert_model')

# Plotting loss and accuracy
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), loss_history, marker='o', color='b', label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), accuracy_history, marker='o', color='g', label='Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()
