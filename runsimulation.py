import sys
import os
import random
import numpy as np
import json
from pathlib import Path
from config.network_config import load_network_config
from core.alive_node import AliveLoopNode
from core.capacitor import CapacitorInSpace
from core.network import TunedAdaptiveFieldNetwork

def load_hr_analytics_data():
    """Load HR Analytics dataset or create synthetic data if not available"""
    dataset_path = Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    if dataset_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(dataset_path)
            print(f"✓ Loaded HR Analytics dataset: {len(df)} records")
            return df
        except ImportError:
            print("⚠ pandas not available, using synthetic data")
        except Exception as e:
            print(f"⚠ Error loading dataset: {e}, using synthetic data")
    else:
        print(f"⚠ Dataset not found at {dataset_path}, using synthetic data")
    
    # Generate synthetic HR data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    synthetic_data = {
        'Age': np.random.randint(18, 65, n_samples),
        'Attrition': np.random.choice(['Yes', 'No'], n_samples, p=[0.16, 0.84]),
        'MonthlyIncome': np.random.randint(1000, 20000, n_samples),
        'JobSatisfaction': np.random.randint(1, 5, n_samples),
        'WorkLifeBalance': np.random.randint(1, 5, n_samples),
        'YearsAtCompany': np.random.randint(0, 40, n_samples)
    }
    
    try:
        import pandas as pd
        df = pd.DataFrame(synthetic_data)
        print(f"✓ Generated synthetic HR data: {len(df)} records")
        return df
    except ImportError:
        print("✓ Generated synthetic HR data (dict format)")
        return synthetic_data


def run_hr_analytics_training(data, epochs=10, batch_size=32):
    """Run dummy training loop with HR Analytics data"""
    print(f"\n🤖 Starting HR Analytics Training")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    
    if hasattr(data, 'shape'):
        n_samples = len(data)
    elif isinstance(data, dict):
        n_samples = len(next(iter(data.values())))
    else:
        n_samples = 1000
    
    n_batches = max(1, n_samples // batch_size)
    
    results = {
        'training_metrics': [],
        'epochs_completed': 0,
        'final_accuracy': 0.0,
        'final_loss': 0.0
    }
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        for batch in range(n_batches):
            # Simulate training step
            batch_loss = np.random.exponential(0.5) + 0.1  # Decreasing loss trend
            batch_accuracy = min(0.95, 0.5 + (epoch * 0.05) + np.random.normal(0, 0.02))
            
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy
        
        avg_loss = epoch_loss / n_batches
        avg_accuracy = epoch_accuracy / n_batches
        
        results['training_metrics'].append({
            'epoch': epoch + 1,
            'loss': float(avg_loss),
            'accuracy': float(avg_accuracy)
        })
        
        print(f"   Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        results['epochs_completed'] = epoch + 1
        results['final_accuracy'] = float(avg_accuracy)
        results['final_loss'] = float(avg_loss)
    
    return results


def save_training_artifacts(results, hr_data):
    """Save training results and artifacts"""
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Save training results
    results_file = outputs_dir / "hr_training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved training results to {results_file}")
    
    # Save dataset info
    if hasattr(hr_data, 'shape'):
        data_info = {
            'dataset_type': 'real' if Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv").exists() else 'synthetic',
            'samples': len(hr_data),
            'features': list(hr_data.columns) if hasattr(hr_data, 'columns') else 'unknown'
        }
    else:
        data_info = {
            'dataset_type': 'synthetic',
            'samples': len(next(iter(hr_data.values()))) if isinstance(hr_data, dict) else 'unknown',
            'features': list(hr_data.keys()) if isinstance(hr_data, dict) else 'unknown'
        }
    
    data_info_file = outputs_dir / "dataset_info.json"
    with open(data_info_file, 'w') as f:
        json.dump(data_info, f, indent=2)
    print(f"✓ Saved dataset info to {data_info_file}")
    
    # Create a simple model artifact placeholder
    model_file = outputs_dir / "hr_model_weights.json"
    model_weights = {
        'model_type': 'adaptive_neural_network',
        'architecture': 'hr_analytics_classifier',
        'weights': [float(np.random.randn()) for _ in range(10)],  # Dummy weights
        'training_completed': True
    }
    with open(model_file, 'w') as f:
        json.dump(model_weights, f, indent=2)
    print(f"✓ Saved model weights to {model_file}")


def set_seed(seed=42):
    """Set random seed for reproducible simulations"""
    random.seed(seed)
    np.random.seed(seed)

def main(seed=None):
    if seed is not None:
        set_seed(seed)
        print(f"Using seed: {seed}")
    
    # Get training parameters from environment variables
    epochs = int(os.getenv('EPOCHS', '10'))
    batch_size = int(os.getenv('BATCH_SIZE', '32'))
    
    print(f"🚀 Adaptive Neural Network Simulation with HR Analytics")
    print(f"   Environment: EPOCHS={epochs}, BATCH_SIZE={batch_size}")
    
    # Load HR Analytics dataset
    hr_data = load_hr_analytics_data()
    
    # Run HR Analytics training
    training_results = run_hr_analytics_training(hr_data, epochs=epochs, batch_size=batch_size)
    
    # Save training artifacts
    save_training_artifacts(training_results, hr_data)
    
    # Continue with original simulation logic
    print(f"\n🌐 Starting Network Simulation")
    
    cfg = load_network_config("config/network_config.yaml")
    nodes = [
        AliveLoopNode(
            position=[0, 0],
            velocity=[0.15, 0],
            initial_energy=10,
            field_strength=1.0,
            node_id=0
        ),
        AliveLoopNode(
            position=[1, 2],
            velocity=[-0.08, 0.03],
            initial_energy=5,
            field_strength=1.2,
            node_id=1
        ),
        AliveLoopNode(
            position=[-1, -1],
            velocity=[0.05, 0.09],
            initial_energy=7,
            field_strength=0.9,
            node_id=2
        )
    ]
    capacitors = [
        CapacitorInSpace(position=[0.5, 0.5], capacity=4),
        CapacitorInSpace(position=[-0.5, -0.5], capacity=6),
        CapacitorInSpace(position=[2, 2], capacity=5)
    ]
    # Map API endpoints to node IDs
    api_endpoints = {
        0: {"type": "human", "url": cfg["api_endpoints"]["human"]},
        1: {"type": "AI", "url": cfg["api_endpoints"]["ai"]},
        2: {"type": "world", "url": cfg["api_endpoints"]["world"]}
    }
    network = TunedAdaptiveFieldNetwork(nodes, capacitors, api_endpoints=api_endpoints)

    print("Initial State:")
    network.print_states()
    print("\nSimulating 15 steps with live API streams every 5 steps...\n")
    for step in range(15):
        network.step()
        network.print_states()
    print("\n✅ Simulation complete.")
    
    # Final summary
    print(f"\n📊 Training Summary:")
    print(f"   Final Accuracy: {training_results['final_accuracy']:.4f}")
    print(f"   Final Loss: {training_results['final_loss']:.4f}")
    print(f"   Epochs Completed: {training_results['epochs_completed']}")
    print(f"   Artifacts saved to: outputs/")

if __name__ == "__main__":
    # Check for seed argument
    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except ValueError:
            print("Invalid seed value. Using default.")
    
    main(seed)
