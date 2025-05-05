import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def plot_training_history(results_file, output_dir):
    """Plot training history metrics."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract metrics
    epochs = range(1, len(results['train_loss']) + 1)
    train_loss = results['train_loss']
    val_accuracy = [metrics['accuracy'] for metrics in results['val_metrics']]
    val_f1 = [metrics['f1_score'] for metrics in results['val_metrics']]
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracy, 'r-', label='Validation Accuracy')
    plt.plot(epochs, val_f1, 'g-', label='Validation F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300)
    plt.close()


def plot_confusion_matrix(results_file, output_dir):
    """Plot the confusion matrix from test results."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    cm = np.array(results['test_metrics']['confusion_matrix'])
    
    # Create normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', 
                xticklabels=['Non-Vulnerable', 'Vulnerable'],
                yticklabels=['Non-Vulnerable', 'Vulnerable'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()


def plot_roc_curve(model, test_loader, device, output_dir):
    """Plot ROC curve for the model on test data."""
    model.eval()
    y_true = []
    y_scores = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            y_true.extend(data.y.cpu().numpy())
            # Get probability for the positive class
            probs = torch.exp(output)[:, 1].cpu().numpy()
            y_scores.extend(probs)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
    plt.close()


def visualize_node_embeddings(model, data_loader, device, output_dir):
    """Visualize node embeddings using t-SNE."""
    model.eval()
    
    # Collect node embeddings and labels
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            
            # Get node embeddings from the GNN's first layer
            x = data.x
            edge_index = data.edge_index
            
            # Apply first layer
            if model.model_type == 'gcn':
                x = model.conv1(x, edge_index)
            elif model.model_type in ['gat', 'gatv2']:
                x = model.conv1(x, edge_index)
            
            x = x.cpu().numpy()
            
            # Get the graph label
            graph_label = data.y.item()
            
            # Store embeddings and labels
            for i in range(x.shape[0]):
                embeddings.append(x[i])
                labels.append(graph_label)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot embeddings
    plt.figure(figsize=(10, 8))
    for label in [0, 1]:
        mask = np.array(labels) == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            label=f"Class {label}", 
            alpha=0.6
        )
    
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'node_embeddings.png'), dpi=300)
    plt.close()


def visualize_graph_with_attention(model, data, original_nx_graph, output_dir, example_idx=0):
    """Visualize a graph with attention weights (for GAT/GATv2 models)."""
    if model.model_type not in ['gat', 'gatv2']:
        print("Attention visualization only works with GAT or GATv2 models")
        return
    
    model.eval()
    
    # Get attention weights
    with torch.no_grad():
        data = data.to(next(model.parameters()).device)
        attention_weights = model.conv1.att_weight  # This might need adaptation based on your GAT implementation
    
    # Convert to networkx graph for visualization
    if not isinstance(original_nx_graph, nx.Graph):
        raise ValueError("original_nx_graph must be a NetworkX graph")
    
    plt.figure(figsize=(15, 12))
    
    # Prepare node colors and sizes
    node_colors = []
    node_sizes = []
    
    for node, attrs in original_nx_graph.nodes(data=True):
        node_type = attrs.get('type', '')
        if 'FunctionDefinition' in node_type:
            node_colors.append('red')
            node_sizes.append(1000)
        elif 'FunctionCall' in node_type:
            node_colors.append('orange')
            node_sizes.append(700)
        elif 'ControlStructure' in node_type:
            node_colors.append('blue')
            node_sizes.append(700)
        elif 'Variable' in node_type:
            node_colors.append('green')
            node_sizes.append(500)
        else:
            node_colors.append('gray')
            node_sizes.append(300)
    
    # Prepare edge colors and widths based on attention weights
    edge_colors = []
    edge_widths = []
    
    # If we have attention weights, use them (simplified example)
    if hasattr(model.conv1, 'att_weight') and model.conv1.att_weight is not None:
        att_weights = model.conv1.att_weight.cpu().numpy()
        
        # Normalize weights for visualization
        if len(att_weights) > 0:
            min_weight = np.min(att_weights)
            max_weight = np.max(att_weights)
            
            # Avoid division by zero
            if max_weight > min_weight:
                norm_weights = (att_weights - min_weight) / (max_weight - min_weight)
            else:
                norm_weights = np.ones_like(att_weights) * 0.5
            
            for e, (u, v) in enumerate(original_nx_graph.edges()):
                if e < len(norm_weights):
                    weight = norm_weights[e]
                    edge_colors.append(plt.cm.viridis(weight))
                    edge_widths.append(1 + 3 * weight)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(1)
        else:
            # If no attention weights, use default colors and widths
            edge_colors = ['gray'] * len(original_nx_graph.edges())
            edge_widths = [1] * len(original_nx_graph.edges())
    else:
        # If no attention weights, use default colors and widths
        edge_colors = ['gray'] * len(original_nx_graph.edges())
        edge_widths = [1] * len(original_nx_graph.edges())
    
    # Create position layout
    pos = nx.spring_layout(original_nx_graph, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(
        original_nx_graph, pos, 
        node_color=node_colors, 
        node_size=node_sizes, 
        alpha=0.8
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        original_nx_graph, pos, 
        edge_color=edge_colors, 
        width=edge_widths, 
        alpha=0.7
    )
    
    # Add labels
    labels = {node: node for node in original_nx_graph.nodes()}
    nx.draw_networkx_labels(original_nx_graph, pos, labels, font_size=10)
    
    # Add title
    plt.title(f"Graph Visualization with Attention - Example {example_idx}")
    plt.axis('off')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'graph_attention_{example_idx}.png'), dpi=300)
    plt.close()


def analyze_feature_importance(model, dataset, device, output_dir, num_samples=10):
    """Analyze feature importance by measuring how changes affect predictions."""
    model.eval()
    
    # Get a subset of graphs for analysis
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Track feature importance scores
    feature_importance = np.zeros(dataset[0].x.shape[1])
    feature_names = [f"Feature_{i}" for i in range(dataset[0].x.shape[1])]
    
    for idx in indices:
        data = dataset[idx].to(device)
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(data)
            original_prob = torch.exp(original_output)[0, 1].item()  # Probability of being vulnerable
        
        # Perturb each feature and measure effect
        for feature_idx in range(data.x.shape[1]):
            # Temporarily change the feature value for all nodes
            modified_data = data.clone()
            modified_data.x[:, feature_idx] = 1 - modified_data.x[:, feature_idx]  # Flip binary features
            
            # Get new prediction
            with torch.no_grad():
                modified_output = model(modified_data)
                modified_prob = torch.exp(modified_output)[0, 1].item()
            
            # Impact is the absolute difference in vulnerability probability
            impact = abs(original_prob - modified_prob)
            feature_importance[feature_idx] += impact
    
    # Average across samples
    feature_importance /= len(indices)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sorted_idx = np.argsort(feature_importance)
    plt.barh([feature_names[i] for i in sorted_idx], feature_importance[sorted_idx])
    plt.title('Feature Importance Analysis')
    plt.xlabel('Average Impact on Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()


def run_visualization(model_path, dataset_path, graph_database_path, output_dir='gnn_visualizations'):
    """Run various visualization analyses for the GNN model."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results_file = os.path.join(os.path.dirname(model_path), 'results.json')
    
    # Plot training history
    if os.path.exists(results_file):
        print("Plotting training history and confusion matrix...")
        plot_training_history(results_file, output_dir)
        plot_confusion_matrix(results_file, output_dir)
    else:
        print(f"Results file not found at {results_file}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Import required modules dynamically to avoid circular imports
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gnn_implementation import CCodeGraphDataset, GNNModel
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = CCodeGraphDataset(root=dataset_path, database_path=graph_database_path)
        
        # Get input dimension from the first graph
        input_dim = dataset[0].x.size(1)
        
        # Load model configuration from results file
        with open(results_file, 'r') as f:
            results = json.load(f)
            model_config = results.get('model_config', {})
            
        # Default values if not in results
        hidden_dim = model_config.get('hidden_dim', 64)
        model_type = model_config.get('model_type', 'gcn')
        dropout = model_config.get('dropout', 0.5)
        
        # Initialize model with same architecture
        print(f"Initializing {model_type.upper()} model...")
        model = GNNModel(
            input_dim=input_dim, 
            hidden_dim=hidden_dim,
            output_dim=2,
            dropout=dropout,
            model_type=model_type
        ).to(device)
        
        # Load model weights
        print(f"Loading model weights from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Create data loaders
        from torch_geometric.data import DataLoader
        test_loader = DataLoader(dataset, batch_size=32)
        
        # Generate visualizations
        print("Generating ROC and PR curves...")
        plot_roc_curve(model, test_loader, device, output_dir)
        
        print("Visualizing node embeddings...")
        visualize_node_embeddings(model, test_loader, device, output_dir)
        
        print("Analyzing feature importance...")
        analyze_feature_importance(model, dataset, device, output_dir, num_samples=10)
        
        # Visualize some example graphs with attention (if GAT model)
        if model_type in ['gat', 'gatv2']:
            print("Generating attention visualizations...")
            for i in range(min(5, len(dataset))):
                # Load original NetworkX graph for visualization
                graphml_path = os.path.join(
                    graph_database_path, 
                    "graphml", 
                    f"function_{dataset.func_ids[i]}.graphml"
                )
                
                if os.path.exists(graphml_path):
                    nx_graph = nx.read_graphml(graphml_path)
                    visualize_graph_with_attention(
                        model, dataset[i], nx_graph, output_dir, example_idx=i
                    )
        
        print(f"All visualizations saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize GNN results")
    parser.add_argument("--model", "-m", required=True, 
                        help="Path to the saved model file")
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to the dataset directory")
    parser.add_argument("--database", "-db", required=True,
                        help="Path to the original graph database")
    parser.add_argument("--output", "-o", default="gnn_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--examples", "-e", type=int, default=5,
                        help="Number of example graphs to visualize")
    
    args = parser.parse_args()
    
    run_visualization(
        model_path=args.model,
        dataset_path=args.dataset,
        graph_database_path=args.database,
        output_dir=args.output
    )
