#!/usr/bin/env python3
"""
End-to-end pipeline for analyzing C functions using Graph Neural Networks.
This script ties together:
1. Graph generation from C code
2. GNN model training
3. Visualization and analysis of results
"""

import os
import argparse
import subprocess
import sys
import json
from datetime import datetime

def run_command(cmd, description):
    """Run a shell command and print its output."""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Command failed with return code {process.returncode}")
        return False
    
    return True

def get_script_dir():
    """Get the directory where the scripts are located."""
    return os.path.dirname(os.path.abspath(__file__))

def run_pipeline(args):
    """Run the full GNN pipeline."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output_dir
    run_dir = os.path.join(output_base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the configuration
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Define directories for each stage
    graph_db_dir = os.path.join(run_dir, "graph_database")
    gnn_results_dir = os.path.join(run_dir, "gnn_results")
    visualization_dir = os.path.join(run_dir, "visualizations")
    
    # Paths to scripts
    script_dir = get_script_dir()
    graph_gen_script = os.path.join(script_dir, "graph_generator.py")
    gnn_train_script = os.path.join(script_dir, "gnn_implementation.py")
    visualization_script = os.path.join(script_dir, "gnn_visualization.py")
    
    # Make scripts executable
    for script in [graph_gen_script, gnn_train_script, visualization_script]:
        os.chmod(script, 0o755)
    
    # Step 1: Generate graph database
    if not args.skip_graph_generation:
        success = run_command(
            [
                "python", graph_gen_script,
                "--input", args.input_json,
                "--output", graph_db_dir,
                "--num-functions", str(args.num_functions) if args.num_functions else "None"
            ],
            "STEP 1: Generating Graph Database from C Functions"
        )
        if not success:
            print("Graph generation failed. Stopping pipeline.")
            return False
    else:
        print("\nSkipping graph generation as requested.")
        graph_db_dir = args.graph_db_dir
        if not os.path.exists(graph_db_dir):
            print(f"Error: Graph database directory {graph_db_dir} not found.")
            return False
    
    # Step 2: Train GNN model
    if not args.skip_training:
        success = run_command(
            [
                "python", gnn_train_script,
                "--database", graph_db_dir,
                "--output", gnn_results_dir,
                "--model", args.model_type,
                "--hidden-dim", str(args.hidden_dim),
                "--learning-rate", str(args.learning_rate),
                "--dropout", str(args.dropout),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size)
            ],
            "STEP 2: Training GNN Model"
        )
        if not success:
            print("GNN training failed. Stopping pipeline.")
            return False
    else:
        print("\nSkipping GNN training as requested.")
        gnn_results_dir = args.gnn_results_dir
        if not os.path.exists(gnn_results_dir):
            print(f"Error: GNN results directory {gnn_results_dir} not found.")
            return False
    
    # Step 3: Generate visualizations
    if not args.skip_visualization:
        model_path = os.path.join(gnn_results_dir, "best_model.pt")
        dataset_path = os.path.join(gnn_results_dir, "dataset")
        
        success = run_command(
            [
                "python", visualization_script,
                "--model", model_path,
                "--dataset", dataset_path,
                "--database", graph_db_dir,
                "--output", visualization_dir
            ],
            "STEP 3: Generating Visualizations and Analysis"
        )
        if not success:
            print("Visualization generation failed.")
            return False
    else:
        print("\nSkipping visualization generation as requested.")
    
    print(f"\n{'='*80}")
    print(f"Pipeline completed successfully!")
    print(f"Results stored in: {run_dir}")
    print(f"{'='*80}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete GNN pipeline for C function vulnerability analysis"
    )
    
    # Input/output options
    parser.add_argument("--input-json", "-i", required=True,
                      help="Input JSON file containing C functions")
    parser.add_argument("--output-dir", "-o", default="gnn_pipeline_output",
                      help="Base output directory for all pipeline results")
    
    # Pipeline control options
    parser.add_argument("--skip-graph-generation", action="store_true",
                      help="Skip graph generation step (requires --graph-db-dir)")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip GNN training step (requires --gnn-results-dir)")
    parser.add_argument("--skip-visualization", action="store_true",
                      help="Skip visualization generation step")
    
    # Parameters for skipping steps
    parser.add_argument("--graph-db-dir", 
                      help="Path to existing graph database (if skipping generation)")
    parser.add_argument("--gnn-results-dir",
                      help="Path to existing GNN results (if skipping training)")
    
    # Graph generation options
    parser.add_argument("--num-functions", "-n", type=int, default=None,
                      help="Number of functions to process (default: all)")
    
    # GNN model options
    parser.add_argument("--model-type", "-m", default="gcn",
                      choices=["gcn", "gat", "gatv2"],
                      help="GNN model type (default: gcn)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                      help="Hidden dimension size (default: 64)")
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001,
                      help="Learning rate (default: 0.001)")
    parser.add_argument("--dropout", type=float, default=0.5,
                      help="Dropout probability (default: 0.5)")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                      help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                      help="Batch size (default: 32)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.skip_graph_generation and not args.graph_db_dir:
        parser.error("--skip-graph-generation requires --graph-db-dir")
    
    if args.skip_training and not args.gnn_results_dir:
        parser.error("--skip-training requires --gnn-results-dir")
    
    # Run the pipeline
    success = run_pipeline(args)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
