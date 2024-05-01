import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification


# Suppress symlink warnings from Hugging Face transformers
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Function to calculate accuracy of predictions
def calculate_accuracy(preds, labels):
    pred_flat = preds.argmax(dim=1).flatten()  # Get the highest probability prediction
    labels_flat = labels.flatten()
    return (pred_flat == labels_flat).cpu().numpy().mean()  # Calculate accuracy


# Data loading and preprocessing
df = pd.read_excel(r'D:\Project\ML_9\InitiativeTracker-main\InitiativeTracker-main\data1.xlsx', engine='openpyxl')
print(df.columns)
keywords = df['Keywords'].tolist()
descriptions = df['Description'].tolist()
classes = df['Class'].tolist()

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
combined_features = [f"{k} {d}" for k, d in zip(keywords, descriptions)]
inputs = tokenizer(combined_features, padding=True, truncation=True, return_tensors="pt", max_length=512)

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(classes)

'''# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using CUDA on device: {device}')
else:
    raise SystemError("CUDA is not available on this system.")'''

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'   # Use CPU for training
#print(torch.cuda.is_available())

# Dataset and DataLoader setup
class DnDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Split the dataset into training and validation sets
dataset = DnDDataset(inputs, y)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Prepare data loaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
validation_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load and configure the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(set(classes)))
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training and validation loops
model.train()
for epoch in range(4):
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    for batch in validation_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            total_eval_accuracy += calculate_accuracy(logits, labels)

    avg_val_accuracy = total_eval_accuracy / len(validation_loader)
    avg_val_loss = total_eval_loss / len(validation_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

# Save trained model and tokenizer
model.save_pretrained(r'D:\Project\ML_9\InitiativeTracker-main\InitiativeTracker-main\bert_model')
tokenizer.save_pretrained(r'D:\Project\ML_9\InitiativeTracker-main\InitiativeTracker-main\bert_model')


