
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import random
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ===== PART 1: Deep Learning Component =====

class CodeTokenizer:
    """Simple tokenizer for code"""
    def __init__(self, max_vocab_size=5000, max_seq_length=256):  # Reduced values for efficiency
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.reverse_vocab = {0: "<PAD>", 1: "<UNK>"}
        self.token_counts = {}
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        
    def tokenize(self, code):
        """Simple tokenization by splitting on whitespace and special characters"""
        # Replace common punctuation with spaces around them for tokenization
        for char in "(){}[];,.:+-*/=&|<>!":
            code = code.replace(char, f" {char} ")
        
        # Split on whitespace
        return code.split()
    
    def build_vocab(self, code_samples):
        """Build vocabulary from code samples"""
        logging.info("Building vocabulary...")
        for code in tqdm(code_samples):
            tokens = self.tokenize(code)
            for token in tokens:
                if token not in self.token_counts:
                    self.token_counts[token] = 0
                self.token_counts[token] += 1
        
        # Sort tokens by frequency
        sorted_tokens = sorted(self.token_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Keep only max_vocab_size tokens
        for i, (token, _) in enumerate(sorted_tokens[:self.max_vocab_size-2]):  # -2 for PAD and UNK
            self.vocab[token] = i + 2  # +2 because PAD=0, UNK=1
            self.reverse_vocab[i + 2] = token
            
        logging.info(f"Vocabulary size: {len(self.vocab)}")
        
    def encode(self, code):
        """Convert code string to token IDs"""
        tokens = self.tokenize(code)
        # Truncate if necessary
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Convert tokens to IDs
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # Pad if necessary
        if len(token_ids) < self.max_seq_length:
            token_ids = token_ids + [self.vocab["<PAD>"]] * (self.max_seq_length - len(token_ids))
            
        return token_ids
    
    def get_vocab_size(self):
        """Return the vocabulary size"""
        return len(self.vocab)


class VulnerabilityDataset(Dataset):
    """Dataset for vulnerability prediction"""
    def __init__(self, codes, targets, tokenizer):
        self.codes = codes
        self.targets = targets
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        code = self.codes[idx]
        target = self.targets[idx]
        
        # Encode the code
        token_ids = self.tokenizer.encode(code)
        
        return {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float)
        }


class CodeEmbeddingModel(nn.Module):
    """Neural network for code embedding - simplified for faster training"""
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1, dropout=0.3):
        super(CodeEmbeddingModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.relu = nn.ReLU()
        
        # Output embeddings instead of predictions for feature extraction
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, token_ids):
        # Get embeddings
        embedded = self.embedding(token_ids)
        
        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Get the final hidden state from both directions
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Pass through fully connected layers
        fc1_out = self.relu(self.fc1(self.dropout(hidden)))
        
        # Get the final embeddings
        embeddings = self.fc_out(self.dropout(fc1_out))
        
        return embeddings

# ===== PART 2: Feature Extraction Component =====

def extract_code_features(func_text):
    """Extract basic code features"""
    features = {}
    
    # Count lines of code
    features['loc'] = len(func_text.split('\n'))
    
    # Count security-related keywords
    security_keywords = ['buffer', 'overflow', 'memory', 'free', 'null', 'check', 
                        'validate', 'size', 'length', 'bounds', 'allocation']
    for keyword in security_keywords:
        features[f'has_{keyword}'] = 1 if keyword in func_text.lower() else 0
    
    # Count function calls
    function_calls = len(re.findall(r'\w+\s*\(', func_text))
    features['function_calls'] = function_calls
    
    # Count pointers and memory operations
    features['pointer_usage'] = func_text.count('*')
    features['address_usage'] = func_text.count('&')
    
    # Count conditions and loops
    features['if_count'] = len(re.findall(r'\bif\s*\(', func_text))
    features['for_count'] = len(re.findall(r'\bfor\s*\(', func_text))
    features['while_count'] = len(re.findall(r'\bwhile\s*\(', func_text))
    
    # Count memory manipulation functions
    mem_funcs = ['malloc', 'calloc', 'realloc', 'free', 'memcpy', 'memset', 'memmove', 'strcpy', 'strncpy']
    for func in mem_funcs:
        features[f'calls_{func}'] = len(re.findall(r'\b' + func + r'\s*\(', func_text))
    
    return features

# ===== PART 3: Main Training and Evaluation Functions =====

def load_jsonl_data(file_path):
    """Load data from a JSONL file into a pandas DataFrame."""
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                logging.warning(f"Error parsing line {line_num}: {e}")
    
    return pd.DataFrame(data)


def train_embedding_model(model, train_loader, val_loader, device, num_epochs=5, learning_rate=0.001):
    """Train the neural network embedding model"""
    # Use MSE loss for learning embeddings
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For tracking metrics - ADD F1 SCORE HERE
    metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],      # Add F1 score tracking
        'val_precision': [],  # Add precision
        'val_recall': []   # Add recall
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train NN]")
        
        for batch in progress_bar:
            token_ids = batch['token_ids'].to(device)
            targets = batch['target'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass - get embeddings
            embeddings = model(token_ids)
            
            # Create pseudo-targets - this is a trick to train the embeddings
            # We want similar codes with same label to have similar embeddings
            target_embeddings = torch.zeros_like(embeddings)
            for i, t in enumerate(targets):
                # Set target embedding based on vulnerability (1) or not (0)
                if t > 0.5:  # Vulnerable
                    target_embeddings[i] = 1.0
                # Non-vulnerable stays at 0
            
            loss = criterion(embeddings, target_embeddings)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_true = []
        val_pred = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val NN]")
            
            for batch in progress_bar:
                token_ids = batch['token_ids'].to(device)
                targets = batch['target']
                
                # Forward pass - get embeddings
                embeddings = model(token_ids)
                
                # Create pseudo-targets
                target_embeddings = torch.zeros_like(embeddings)
                for i, t in enumerate(targets):
                    if t > 0.5:  # Vulnerable
                        target_embeddings[i] = 1.0
                
                loss = criterion(embeddings, target_embeddings)
                
                # Calculate predictions for F1 score - use mean embedding value > 0.5 as prediction
                pred = (torch.mean(embeddings, dim=1) > 0.5).float().cpu().numpy()
                true = targets.numpy()
                
                # Collect predictions and true values
                val_true.extend(true)
                val_pred.extend(pred)
                
                val_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
        
        val_loss /= len(val_loader)
        
        # Calculate F1 score, precision and recall
        val_f1 = f1_score(val_true, val_pred, zero_division=0)
        val_precision = precision_score(val_true, val_pred, zero_division=0)
        val_recall = recall_score(val_true, val_pred, zero_division=0)
        
        # Save metrics
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_f1'].append(val_f1)
        metrics['val_precision'].append(val_precision)
        metrics['val_recall'].append(val_recall)
        
        # Save metrics to CSV after each epoch for later analysis
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(metrics).to_csv(os.path.join("results", 'nn_training_metrics.csv'), index=False)
        
        logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                     f"Train Loss: {train_loss:.4f}, "
                     f"Val Loss: {val_loss:.4f}, "
                     f"Val F1: {val_f1:.4f}")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['epoch'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neural Network Training Loss')
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(os.path.join("results", 'nn_training_loss.png'))
    
    # Plot F1 score
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['epoch'], metrics['val_f1'], label='F1 Score', marker='o')
    plt.plot(metrics['epoch'], metrics['val_precision'], label='Precision', marker='s')
    plt.plot(metrics['epoch'], metrics['val_recall'], label='Recall', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Neural Network Validation Metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", 'nn_validation_metrics.png'))
    
    return model, metrics

def extract_embeddings(model, data_loader, device):
    """Extract embeddings from the trained model"""
    model.eval()
    all_embeddings = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            token_ids = batch['token_ids'].to(device)
            targets = batch['target']
            
            # Forward pass to get embeddings
            embeddings = model(token_ids)
            
            # Store embeddings and targets
            all_embeddings.append(embeddings.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    return all_embeddings, np.array(all_targets)


def create_balanced_dataset(file_path, balance_ratio=1.0, max_samples=5000):
    """Create balanced dataset from JSONL file"""
    vulnerable_samples = []
    non_vulnerable_samples = []
    
    logging.info(f"Creating balanced dataset from {file_path}")
    
    # Load data
    df = load_jsonl_data(file_path)
    
    # Separate by target
    for _, row in df.iterrows():
        if row['target'] == 1:
            vulnerable_samples.append(row)
        else:
            non_vulnerable_samples.append(row)
    
    num_vulnerable = len(vulnerable_samples)
    num_non_vulnerable = len(non_vulnerable_samples)
    
    logging.info(f"Original dataset - Vulnerable: {num_vulnerable}, Non-vulnerable: {num_non_vulnerable}")
    
    # Balance the dataset
    if num_vulnerable < num_non_vulnerable:
        num_non_vulnerable_to_keep = min(int(num_vulnerable * balance_ratio), num_non_vulnerable)
        non_vulnerable_samples = random.sample(non_vulnerable_samples, num_non_vulnerable_to_keep)
    else:
        num_vulnerable_to_keep = min(int(num_non_vulnerable * balance_ratio), num_vulnerable)
        vulnerable_samples = random.sample(vulnerable_samples, num_vulnerable_to_keep)
    
    # Combine and shuffle
    balanced_samples = vulnerable_samples + non_vulnerable_samples
    
    # Limit the total dataset size if needed
    if max_samples and len(balanced_samples) > max_samples:
        balanced_samples = random.sample(balanced_samples, max_samples)
    
    # Shuffle the results
    random.shuffle(balanced_samples)
    
    balanced_df = pd.DataFrame(balanced_samples)
    
    logging.info(f"Balanced dataset - Target distribution: {balanced_df['target'].value_counts().to_dict()}")
    logging.info(f"Total samples: {len(balanced_df)}")
    
    return balanced_df


def main(file_path, use_balanced_dataset=True, max_samples=5000):
    """Main function to run the hybrid model"""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load and balance data
    if use_balanced_dataset:
        df = create_balanced_dataset(file_path, balance_ratio=1.0, max_samples=max_samples)
    else:
        # Load original data
        logging.info(f"Loading data from {file_path}")
        df = load_jsonl_data(file_path)
    
    if df.empty:
        logging.error("No data loaded. Exiting.")
        return
    
    logging.info(f"Loaded {len(df)} records")
    
    # Display data summary
    logging.info(f"Data columns: {df.columns.tolist()}")
    if 'target' in df.columns:
        logging.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # Get data
    codes = df['func'].tolist()
    targets = df['target'].astype(int).tolist() if 'target' in df.columns else None
    
    if targets is None:
        logging.error("No target column found. Exiting.")
        return
    
    # Split into training and validation sets
    train_codes, val_codes, train_targets, val_targets = train_test_split(
        codes, targets, test_size=0.2, random_state=42, stratify=targets
    )
    
    # STEP 1: Train the neural network embedding model
    logging.info("Step 1: Training neural network embedding model...")
    
    # Initialize tokenizer and build vocabulary
    tokenizer = CodeTokenizer(max_vocab_size=5000, max_seq_length=256)  # Reduced for efficiency
    tokenizer.build_vocab(train_codes)
    
    # Create datasets
    train_dataset = VulnerabilityDataset(train_codes, train_targets, tokenizer)
    val_dataset = VulnerabilityDataset(val_codes, val_targets, tokenizer)
    
    # Create data loaders with larger batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Initialize embedding model with smaller dimensions
    embedding_model = CodeEmbeddingModel(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=64,  
        hidden_dim=128,   
        num_layers=1,     
        dropout=0.3
    ).to(device)
    
    # Train embedding model with fewer epochs
    embedding_model = train_embedding_model(
        model=embedding_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=3, 
        learning_rate=0.001
    )
    
    # Save embedding model
    torch.save(embedding_model.state_dict(), 'embedding_model.pth')
    
    # STEP 2: Extract embeddings and traditional features
    logging.info("Step 2: Extracting features for gradient boosting...")
    
    # Extract embeddings
    train_embeddings, _ = extract_embeddings(embedding_model, train_loader, device)
    val_embeddings, _ = extract_embeddings(embedding_model, val_loader, device)
    
    logging.info(f"Extracted embeddings shape: {train_embeddings.shape}")
    
    # Create column names for embeddings
    embedding_cols = [f'emb_{i}' for i in range(train_embeddings.shape[1])]
    
    # Create DataFrames with embeddings
    train_emb_df = pd.DataFrame(train_embeddings, columns=embedding_cols)
    val_emb_df = pd.DataFrame(val_embeddings, columns=embedding_cols)
    
    # Extract traditional features
    logging.info("Extracting traditional code features...")
    train_features = [extract_code_features(code) for code in tqdm(train_codes)]
    val_features = [extract_code_features(code) for code in tqdm(val_codes)]
    
    train_features_df = pd.DataFrame(train_features)
    val_features_df = pd.DataFrame(val_features)
    
    # Extract TF-IDF features - reduced max_features for efficiency
    logging.info("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')  # Reduced from 200
    
    train_tfidf = tfidf.fit_transform(train_codes)
    val_tfidf = tfidf.transform(val_codes)
    
    train_tfidf_df = pd.DataFrame(
        train_tfidf.toarray(), 
        columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    )
    
    val_tfidf_df = pd.DataFrame(
        val_tfidf.toarray(), 
        columns=[f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    )
    
    # Combine all features for gradient boosting
    train_combined = pd.concat([train_emb_df, train_features_df, train_tfidf_df], axis=1)
    val_combined = pd.concat([val_emb_df, val_features_df, val_tfidf_df], axis=1)
    
    logging.info(f"Combined features shape: {train_combined.shape}")
    
    # STEP 3: Train gradient boosting model on the combined features
    logging.info("Step 3: Training gradient boosting model...")
    
    # Initialize XGBoost model with fewer estimators for faster training
    xgb_model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=4,        # Reduced from 5
        n_estimators=50,    # Reduced from 100
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    # Train XGBoost model
    xgb_model.fit(
        train_combined, 
        train_targets, 
        eval_set=[(val_combined, val_targets)], 
        early_stopping_rounds=5,  # Reduced from 10
        verbose=False
    )
    
    # Save gradient boosting model
    xgb_model.save_model('xgb_model.json')
    
    # STEP 4: Final evaluation
    logging.info("Step 4: Final model evaluation...")
    
    # Make predictions
    val_preds = xgb_model.predict(val_combined)
    val_probs = xgb_model.predict_proba(val_combined)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(val_targets, val_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(val_targets, val_preds, average='binary')
    
    logging.info(f"Final model performance:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    
    # Print classification report
    logging.info("\nClassification Report:\n" + classification_report(val_targets, val_preds))
    
    # Feature importance
    feature_importances = xgb_model.feature_importances_
    feature_names = train_combined.columns
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # Sort by importance and get top features
    importance_df = importance_df.sort_values('Importance', ascending=False)
    top_features = importance_df.head(20)
    
    logging.info("\nTop 20 Important Features:\n" + str(top_features))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'][:10], top_features['Importance'][:10])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return embedding_model, xgb_model, tokenizer


class HybridVulnerabilityPredictor:
    """Class for making predictions with the hybrid model"""
    def __init__(self, embedding_model, xgb_model, tokenizer, device):
        self.embedding_model = embedding_model
        self.xgb_model = xgb_model
        self.tokenizer = tokenizer
        self.device = device
        
    def extract_features(self, code):
        """Extract all features for a single code sample"""
        # Get code embeddings
        token_ids = torch.tensor([self.tokenizer.encode(code)], dtype=torch.long).to(self.device)
        self.embedding_model.eval()
        with torch.no_grad():
            embedding = self.embedding_model(token_ids).cpu().numpy()[0]
        
        # Get traditional features
        trad_features = extract_code_features(code)
        
        # Create DataFrame with all features
        emb_df = pd.DataFrame([embedding], columns=[f'emb_{i}' for i in range(len(embedding))])
        trad_df = pd.DataFrame([trad_features])
        
        # For TF-IDF, we would need the fitted vectorizer
        # For simplicity, we'll skip TF-IDF in the prediction function
        
        # Combine features
        combined = pd.concat([emb_df, trad_df], axis=1)
        
        return combined
    
    def predict(self, code):
        """Make a prediction for a single code sample"""
        features = self.extract_features(code)
        
        # Get prediction and probability
        prediction = self.xgb_model.predict(features)[0]
        probability = self.xgb_model.predict_proba(features)[0, 1]
        
        return prediction, probability


if __name__ == "__main__":
    file_path = "target_test.json"  # Your input file path
    # Run with balanced dataset and reduced sample size
    main(file_path, use_balanced_dataset=True, max_samples=5000)
