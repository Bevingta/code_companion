import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

# Load CodeT5 model and tokenizer
# Could potentially experiement with other models from the T5 family
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Load dataset
df = pd.read_csv("output.csv")
df = df.dropna()  # Remove NaN values
df = df.astype({"cvss_score": float})  # Ensure scores are float

# Check if the dataset loaded correctly
print(f"Dataset Loaded: {len(df)} entries")

# Tokenization function
def tokenize_function(code):
    return tokenizer(code, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

# Custom PyTorch dataset class
class CodeT5Dataset(Dataset):
    def __init__(self, df):
        self.codes = df["func"].tolist()
        self.scores = df["cvss_score"].tolist()
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        encoded = tokenize_function(self.codes[idx])
        return {
            "input_ids": encoded["input_ids"].squeeze(0), 
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.scores[idx], dtype=torch.float32)
        }

# Prepare dataset and dataloader
dataset = CodeT5Dataset(df)
batch_size = 8
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
# other effective loss functions are SmoothL1Loss() & MSEloss()
loss_fn = nn.HuberLoss(delta=1.0)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop with loss logging
epochs = 5
final_loss = None

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs} ----------------------")
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs.logits.squeeze(-1)  # Extract raw model output


        # Compute loss
        loss = loss_fn(predictions, labels)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for every batch (Loss Per Run)
        print(f"Step {step + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    # Print average loss for the epoch
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")
    final_loss = avg_loss

# Save the fine-tuned model
model.save_pretrained("./codet5_cvss_model")
tokenizer.save_pretrained("./codet5_cvss_model")

# Print final loss after training
print(f"\nTraining complete. Model saved to './codet5_cvss_model'. Final Loss: {final_loss:.4f}")