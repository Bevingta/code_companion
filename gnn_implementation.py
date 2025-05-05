import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time
from tqdm import tqdm

# Import PyG classes needed for serialization
from torch_geometric.data.data import DataTensorAttr, DataEdgeAttr, GlobalStorage

# Add these classes to PyTorch's safe globals list
torch.serialization.add_safe_globals([Data, DataTensorAttr, DataEdgeAttr])
torch.serialization.add_safe_globals([GlobalStorage])

class CCodeGraphDataset(Dataset):
    """Dataset for C code function graphs."""
    
    def __init__(self, root, database_path, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            root: Root directory where the dataset should be saved
            database_path: Path to the graph database directory
            transform, pre_transform, pre_filter: PyG dataset parameters
        """
        self.database_path = database_path
        self.graphml_dir = os.path.join(database_path, "graphml")
        self.mapping_file = os.path.join(database_path, "graph_mapping.json")
        self.stats_file = os.path.join(database_path, "graph_stats.json")
        
        print(f"Loading mapping from: {self.mapping_file}")
        # Load mapping and stats
        try:
            with open(self.mapping_file, 'r') as f:
                self.mapping = json.load(f)
            
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
            
            # Create an index mapping for quick access
            self.func_ids = list(self.mapping['function_to_graph'].keys())
            
            # Create labels (vulnerable or not)
            self.labels = {}
            self.vulnerable_funcs = set(str(func_id) for func_id in self.mapping.get('vulnerable_functions', []))
            print("Vulnerable_funcs: ", self.vulnerable_funcs)
            time.sleep(4)
            for func_id in self.func_ids:
                self.labels[func_id] = 1 if func_id in self.vulnerable_funcs else 0
                
            print(f"Loaded {len(self.func_ids)} function mappings")
        except Exception as e:
            print(f"Error loading mappings: {e}")
            raise
        
        # Call the parent constructor AFTER initializing our attributes
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # Verify that processing completed successfully
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
            
        processed_files = [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
        if len(processed_files) == 0:
            print("No processed files found. Running processing...")
            self.process()
    
    @property
    def raw_file_names(self):
        return [f"function_{func_id}.graphml" for func_id in self.func_ids]
    
    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.func_ids))]
    
    def download(self):
        # No download needed as we're using local files
        pass
    
    def process(self):
        print(f"Processing {len(self.func_ids)} graphs...")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # First pass: collect all node types across all graphs
        all_node_types = set()
        for i, func_id in enumerate(self.func_ids):
            try:
                graphml_path = os.path.join(self.graphml_dir, f"function_{func_id}.graphml")
                
                if not os.path.exists(graphml_path):
                    continue
                    
                try:
                    nx_graph = nx.read_graphml(graphml_path)
                except Exception as e:
                    continue
                
                # Collect node types
                for _, attrs in nx_graph.nodes(data=True):
                    node_type = attrs.get('type', 'unknown')
                    all_node_types.add(node_type)
                    
            except Exception as e:
                pass
        
        # Create a global mapping of node types to indices
        global_node_type_to_idx = {node_type: i for i, node_type in enumerate(all_node_types)}
        print(f"Total unique node types across all graphs: {len(global_node_type_to_idx)}")
        
        # Second pass: create PyG data objects with consistent feature dimensions
        for i, func_id in enumerate(self.func_ids):
            try:
                # Load graph from GraphML
                graphml_path = os.path.join(self.graphml_dir, f"function_{func_id}.graphml")
                
                if not os.path.exists(graphml_path):
                    print(f"WARNING: Graph file not found: {graphml_path}")
                    continue
                    
                try:
                    nx_graph = nx.read_graphml(graphml_path)
                except Exception as e:
                    print(f"Error reading GraphML file {graphml_path}: {e}")
                    continue
                
                # Create feature matrix with global node type mapping
                x = torch.zeros((nx_graph.number_of_nodes(), len(global_node_type_to_idx)), dtype=torch.float)
                node_to_idx = {}
                
                for idx, (node, attrs) in enumerate(nx_graph.nodes(data=True)):
                    node_to_idx[node] = idx
                    node_type = attrs.get('type', 'unknown')
                    x[idx, global_node_type_to_idx[node_type]] = 1.0
                
                # Create edge index
                edge_index = []
                for u, v in nx_graph.edges():
                    # Skip if node indices are not found
                    if u not in node_to_idx or v not in node_to_idx:
                        continue
                    edge_index.append([node_to_idx[u], node_to_idx[v]])
                
                # Skip graphs with no edges
                if len(edge_index) == 0:
                    print(f"WARNING: Graph {func_id} has no edges, skipping")
                    continue
                    
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                
                # Create label (vulnerable or not)
                y = torch.tensor([self.labels[func_id]], dtype=torch.long)
                
                # Create Data object
                data = Data(x=x, edge_index=edge_index, y=y)
                
                # Save
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(self.func_ids)} graphs")
                    
            except Exception as e:
                print(f"Error processing graph {func_id}: {e}")
                import traceback
                traceback.print_exc()
        
        print("Processing complete!") 
    
    def len(self):
        return len(self.func_ids)
    
    def get(self, idx):
        try:
            data_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Processed file not found: {data_path}")
            
            # Use the safe_globals context manager to load PyG data objects
            with torch.serialization.safe_globals([Data, DataTensorAttr, DataEdgeAttr]):
                data = torch.load(data_path)
            
            # Add extra debugging info
            if hasattr(data, 'x') and data.x is not None:
                print(f"Graph {idx} has feature dimension: {data.x.shape[1]}")
            
            return data
        except Exception as e:
            print(f"Error loading data_{idx}.pt: {e}")
            # Return a dummy data object as fallback
            # This is not ideal but prevents crashes during training
            x = torch.zeros((1, 1), dtype=torch.float)
            edge_index = torch.zeros((2, 1), dtype=torch.long)
            y = torch.tensor([0], dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y) 
class GNNModel(nn.Module):
    """Graph Neural Network for vulnerability detection in C code."""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=2, dropout=0.5, model_type='gcn'):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (2 for binary classification)
            dropout: Dropout probability
            model_type: 'gcn', 'gat', or 'gatv2'
        """
        super(GNNModel, self).__init__()
        self.model_type = model_type
        
        # First layer
        if model_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
        elif model_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=8, dropout=dropout)
            hidden_dim = hidden_dim * 8  # GAT concatenates heads by default
        elif model_type == 'gatv2':
            self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=8, dropout=dropout)
            hidden_dim = hidden_dim * 8  # GAT concatenates heads by default
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Second layer
        if model_type == 'gcn':
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif model_type == 'gat':
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        elif model_type == 'gatv2':
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=1, dropout=dropout)
        
        # Output layer
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)


def train_gnn(model, train_loader, optimizer, device):
    """Train the GNN model for one epoch."""
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate_gnn(model, loader, device):
    """Evaluate the GNN model."""
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(dim=1)[1]
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            correct += pred.eq(data.y).sum().item()
            total += data.y.size(0)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
    
    return metrics


def main(database_path, output_dir="gnn_results", model_type="gcn", hidden_dim=64, 
         learning_rate=0.001, dropout=0.5, epochs=100, batch_size=32):
    """Run the GNN training and evaluation pipeline."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = CCodeGraphDataset(root=os.path.join(output_dir, "dataset"), 
                               database_path=database_path) #TODO: ADD A SHUFFLE MECHANIC
    
    # Check for class imbalance
    labels = torch.cat([data.y for data in dataset])
    print(f"Class distribution: {torch.bincount(labels)}")

    # Shuffle the indices first
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    # Split dataset with shuffled indices
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx])

    
    '''# Split dataset
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, stratify=labels[train_idx])'''
    
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get input dimension from the first graph
    input_dim = dataset[0].x.size(1)
    
    # Initialize model
    model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim, 
                    dropout=dropout, model_type=model_type).to(device)
    print(model)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    results = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_gnn(model, train_loader, optimizer, device)
        
        # Evaluate on validation set
        val_metrics = evaluate_gnn(model, val_loader, device)
        
        # Save results
        results['train_loss'].append(train_loss)
        results['val_metrics'].append(val_metrics)
        
        # Print progress
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
              f'Val Acc: {val_metrics["accuracy"]:.4f}, Val F1: {val_metrics["f1_score"]:.4f}')
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
    test_metrics = evaluate_gnn(model, test_loader, device)
    
    print(f'\nTest Results - Best model from epoch {best_epoch}:')
    print(f'Test Accuracy: {test_metrics["accuracy"]:.4f}')
    print(f'Test Precision: {test_metrics["precision"]:.4f}')
    print(f'Test Recall: {test_metrics["recall"]:.4f}')
    print(f'Test F1 Score: {test_metrics["f1_score"]:.4f}')
    print(f'Confusion Matrix:\n{test_metrics["confusion_matrix"]}')
    
    # Save results
    results['test_metrics'] = test_metrics
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key in results['test_metrics']:
            if isinstance(results['test_metrics'][key], np.ndarray):
                results['test_metrics'][key] = results['test_metrics'][key].tolist()
        
        for i, metrics in enumerate(results['val_metrics']):
            for key in metrics:
                if isinstance(metrics[key], np.ndarray):
                    results['val_metrics'][i][key] = metrics[key].tolist()
        
        json.dump(results, f, indent=2)
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GNN on C code graphs")
    parser.add_argument("--database", "-d", default="graph_database", 
                        help="Path to graph database directory")
    parser.add_argument("--output", "-o", default="gnn_results", 
                        help="Output directory for results")
    parser.add_argument("--model", "-m", default="gcn", choices=["gcn", "gat", "gatv2"],
                        help="GNN model type")
    parser.add_argument("--hidden-dim", type=int, default=64, 
                        help="Hidden dimension size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, 
                        help="Dropout probability")
    parser.add_argument("--epochs", "-e", type=int, default=100, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32, 
                        help="Batch size")
    
    args = parser.parse_args()
    
    main(
        database_path=args.database,
        output_dir=args.output,
        model_type=args.model,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
