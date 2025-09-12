"""
Continual learning utilities for adaptive neural networks.

This module provides stubs for future continual learning implementations
including Split MNIST and other sequential learning benchmarks.
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ..api.config import AdaptiveConfig
from ..api.model import AdaptiveModel


def split_mnist_benchmark(
    model: AdaptiveModel, config: AdaptiveConfig, num_tasks: int = 5, use_synthetic: bool = False
) -> dict[str, Any]:
    """
    Split MNIST continual learning benchmark.

    This function implements the Split MNIST benchmark where the
    10 digit classes are split into sequential tasks.

    Args:
        model: Adaptive neural network model
        config: Model configuration
        num_tasks: Number of tasks to split MNIST into
        use_synthetic: Use synthetic data instead of real MNIST (for testing)

    Returns:
        Results dictionary with per-task metrics
    """
    device = torch.device(config.device)
    
    if use_synthetic:
        # Create synthetic MNIST-like data for testing
        def create_synthetic_mnist_data(num_samples: int = 1000):
            # Create synthetic 28x28 images with distinct patterns for each digit
            data_list = []
            labels_list = []
            
            for digit in range(10):
                # Create digit-specific patterns
                samples = torch.randn(num_samples // 10, 784)
                
                # Add digit-specific pattern (simple approach)
                if digit < 5:
                    # First half digits: pattern in first half of image
                    samples[:, :392] += (digit + 1) * 0.5
                else:
                    # Second half digits: pattern in second half of image  
                    samples[:, 392:] += (digit - 4) * 0.5
                
                data_list.append(samples)
                labels_list.extend([digit] * (num_samples // 10))
            
            all_data = torch.cat(data_list, dim=0)
            all_labels = torch.tensor(labels_list)
            
            # Normalize
            all_data = (all_data - all_data.mean()) / (all_data.std() + 1e-8)
            
            return torch.utils.data.TensorDataset(all_data, all_labels)
        
        train_dataset = create_synthetic_mnist_data(800)
        test_dataset = create_synthetic_mnist_data(200)
    else:
        # Load real MNIST dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.flatten())
        ])
        
        train_dataset = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
    
    # Split classes into tasks
    classes_per_task = 10 // num_tasks
    task_results = {}
    
    # Track forgetting - store performance on previous tasks
    task_accuracies = []
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for task_id in range(num_tasks):
        print(f"\n=== Task {task_id + 1}/{num_tasks} ===")
        
        # Define classes for this task
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        task_classes = list(range(start_class, end_class))
        
        # Create task-specific datasets
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in task_classes]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in task_classes]
        
        task_train_dataset = Subset(train_dataset, train_indices)
        task_test_dataset = Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(task_train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(task_test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Train on current task
        task_losses = []
        epochs_per_task = max(1, config.num_epochs // num_tasks)  # Ensure at least 1 epoch
        
        for epoch in range(epochs_per_task):
            epoch_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Map target to task-specific classes (0 to classes_per_task-1)
                target = target - start_class
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Clear any stored computation graphs without fully resetting state
                with torch.no_grad():
                    # Detach tensors to break computation graphs
                    model.node_state.hidden_state = model.node_state.hidden_state.detach()
                    model.node_state.energy = model.node_state.energy.detach()
                    model.node_state.activity = model.node_state.activity.detach()
                
                output = model(data)
                
                # Only use relevant output dimensions for this task
                task_output = output[:, start_class:end_class]
                loss = F.cross_entropy(task_output, target)
                
                # Detach loss to avoid graph issues
                loss_item = loss.item()
                
                loss.backward()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(loss_item)
                
                if batch_idx % (len(train_loader) // 4 + 1) == 0:
                    print(f'Task {task_id+1}, Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss_item:.6f}')
            
            if epoch_losses:  # Only append if we have losses
                task_losses.append(sum(epoch_losses) / len(epoch_losses))
        
        # Test current task performance
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = target - start_class
                output = model(data)
                task_output = output[:, start_class:end_class]
                pred = task_output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        current_acc = correct / total if total > 0 else 0.0
        print(f'Task {task_id+1} Accuracy: {current_acc:.4f}')
        
        # Measure forgetting on previous tasks
        previous_task_accs = []
        for prev_task_id in range(task_id):
            prev_start = prev_task_id * classes_per_task
            prev_end = prev_start + classes_per_task
            prev_classes = list(range(prev_start, prev_end))
            
            prev_test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in prev_classes]
            if not prev_test_indices:  # Skip if no data
                continue
                
            prev_test_dataset = Subset(test_dataset, prev_test_indices)
            prev_test_loader = DataLoader(prev_test_dataset, batch_size=config.batch_size, shuffle=False)
            
            prev_correct = 0
            prev_total = 0
            with torch.no_grad():
                for data, target in prev_test_loader:
                    data, target = data.to(device), target.to(device)
                    target = target - prev_start
                    output = model(data)
                    prev_task_output = output[:, prev_start:prev_end]
                    pred = prev_task_output.argmax(dim=1)
                    prev_correct += (pred == target).sum().item()
                    prev_total += target.size(0)
            
            prev_acc = prev_correct / prev_total if prev_total > 0 else 0.0
            previous_task_accs.append(prev_acc)
        
        # Calculate forgetting metric
        if task_id > 0 and len(previous_task_accs) > 0:
            forgetting_scores = []
            for i, prev_acc in enumerate(previous_task_accs):
                if i < len(task_accuracies):
                    original_acc = task_accuracies[i]
                    forgetting = max(0, original_acc - prev_acc)
                    forgetting_scores.append(forgetting)
            avg_forgetting = sum(forgetting_scores) / len(forgetting_scores) if forgetting_scores else 0.0
        else:
            avg_forgetting = 0.0
        
        # Store results for this task
        task_results[f'task_{task_id + 1}'] = {
            'classes': task_classes,
            'accuracy': current_acc,
            'loss_history': task_losses,
            'previous_task_accuracies': previous_task_accs,
            'average_forgetting': avg_forgetting
        }
        
        task_accuracies.append(current_acc)
        model.train()
    
    # Calculate overall metrics
    final_avg_accuracy = sum(task_accuracies) / len(task_accuracies) if task_accuracies else 0.0
    total_forgetting = sum(result['average_forgetting'] for result in task_results.values()) / num_tasks
    
    return {
        'task_results': task_results,
        'final_average_accuracy': final_avg_accuracy,
        'total_forgetting': total_forgetting,
        'num_tasks': num_tasks,
        'classes_per_task': classes_per_task
    }


def domain_shift_evaluation(
    model: AdaptiveModel, source_loader: DataLoader, target_loaders: list[DataLoader]
) -> dict[str, Any]:
    """
    Placeholder for domain shift robustness evaluation.

    This function will evaluate model robustness to domain shifts
    using corrupted datasets.

    Args:
        model: Trained adaptive neural network model
        source_loader: Original training domain data
        target_loaders: List of shifted domain data loaders

    Returns:
        Results dictionary with robustness metrics

    Raises:
        NotImplementedError: This is a placeholder for future implementation
    """
    raise NotImplementedError(
        "Domain shift evaluation will be implemented in version 0.3.0. "
        "This includes:\n"
        "- CIFAR-10 corrupted datasets (noise, blur, weather, digital)\n"
        "- Robustness metrics and adaptation measurement\n"
        "- Energy-based adaptation strategies\n"
        "- Phase-dependent robustness analysis"
    )


def ablation_study_sleep_phases(
    config: AdaptiveConfig, disable_phases: list[str] | None = None
) -> dict[str, Any]:
    """
    Sleep phase ablation studies.

    This function systematically disables different phases
    to understand their contribution to learning and adaptation.

    Args:
        config: Base model configuration
        disable_phases: List of phases to disable ('sleep', 'interactive', 'inspired')

    Returns:
        Results comparing performance with/without specific phases
    """
    from ..api.model import AdaptiveModel
    # Note: Using basic training loop instead of TrainingLoop class for simplicity
    
    disable_phases = disable_phases or []
    
    # Create synthetic task for evaluation
    def create_synthetic_dataset(samples_per_class: int = 100):
        """Create a simple synthetic dataset for testing."""
        # Create synthetic data matching the output dimension
        num_classes = config.output_dim
        input_dim = config.input_dim
        
        data = []
        labels = []
        
        for class_id in range(num_classes):
            # Create distinct patterns for each class
            class_data = torch.randn(samples_per_class, input_dim)
            
            # Add class-specific patterns to make classes distinguishable
            pattern_start = (class_id * input_dim // num_classes)
            pattern_end = ((class_id + 1) * input_dim // num_classes)
            class_data[:, pattern_start:pattern_end] += (class_id + 1) * 0.5
            
            data.append(class_data)
            labels.extend([class_id] * samples_per_class)
        
        data = torch.cat(data, dim=0)
        labels = torch.tensor(labels)
        
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(data, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    results = {}
    
    # Test different phase configurations
    phase_configs = [
        {'name': 'full', 'disabled': []},
        {'name': 'no_sleep', 'disabled': ['sleep']},
        {'name': 'no_interactive', 'disabled': ['interactive']}, 
        {'name': 'no_inspired', 'disabled': ['inspired']},
        {'name': 'only_active', 'disabled': ['sleep', 'interactive', 'inspired']},
        {'name': 'custom', 'disabled': disable_phases}
    ]
    
    for phase_config in phase_configs:
        print(f"\n=== Testing configuration: {phase_config['name']} ===")
        
        # Create model with modified phase scheduler
        model = AdaptiveModel(config)
        
        # Modify the phase scheduler to disable certain phases
        disabled_phases = phase_config['disabled']
        if disabled_phases:
            # Override the phase transition logic
            original_step = model.phase_scheduler.step
            
            def modified_step(energy_levels, activity_levels, anxiety_levels=None):
                phases = original_step(energy_levels, activity_levels, anxiety_levels)
                
                # Clone phases to avoid warnings about expanded tensors
                phases = phases.clone()
                
                # Replace disabled phases with ACTIVE
                for phase_name in disabled_phases:
                    if phase_name == 'sleep':
                        phases[phases == 1] = 0  # SLEEP -> ACTIVE
                    elif phase_name == 'interactive':
                        phases[phases == 2] = 0  # INTERACTIVE -> ACTIVE  
                    elif phase_name == 'inspired':
                        phases[phases == 3] = 0  # INSPIRED -> ACTIVE
                
                return phases
            
            model.phase_scheduler.step = modified_step
        
        # Train on synthetic task
        train_loader = create_synthetic_dataset()
        test_loader = create_synthetic_dataset(samples_per_class=50)  # Smaller test set
        
        # Create trainer
        # Note: Using basic training loop instead of TrainingLoop class for simplicity
        
        # Track metrics during training
        training_metrics = []
        energy_efficiency = []
        phase_distributions = []
        
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Short training loop for ablation study
        num_epochs = min(5, config.num_epochs)
        for epoch in range(num_epochs):
            epoch_losses = []
            epoch_accuracies = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(config.device), target.to(config.device)
                
                optimizer.zero_grad()
                
                # Clear any stored computation graphs without fully resetting state
                with torch.no_grad():
                    # Detach tensors to break computation graphs
                    if hasattr(model.node_state, 'hidden_state'):
                        model.node_state.hidden_state = model.node_state.hidden_state.detach()
                        model.node_state.energy = model.node_state.energy.detach()
                        model.node_state.activity = model.node_state.activity.detach()
                
                output = model(data)
                loss = F.cross_entropy(output, target)
                
                # Store loss value before backward
                loss_item = loss.item()
                
                loss.backward()
                
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                acc = (pred == target).float().mean()
                
                epoch_losses.append(loss_item)
                epoch_accuracies.append(acc.item())
                
                # Track energy metrics
                if hasattr(model.node_state, 'energy'):
                    mean_energy = model.node_state.energy.mean().item()
                    energy_efficiency.append(mean_energy)
                
                # Track phase distribution
                current_phases = model.phase_scheduler.node_phases
                phase_dist = {
                    'active': (current_phases == 0).float().mean().item(),
                    'sleep': (current_phases == 1).float().mean().item(), 
                    'interactive': (current_phases == 2).float().mean().item(),
                    'inspired': (current_phases == 3).float().mean().item()
                }
                phase_distributions.append(phase_dist)
            
            epoch_metrics = {
                'epoch': epoch,
                'loss': sum(epoch_losses) / len(epoch_losses),
                'accuracy': sum(epoch_accuracies) / len(epoch_accuracies)
            }
            training_metrics.append(epoch_metrics)
            print(f"Epoch {epoch+1}: Loss={epoch_metrics['loss']:.4f}, Acc={epoch_metrics['accuracy']:.4f}")
        
        # Evaluate final performance
        model.eval()
        test_correct = 0
        test_total = 0
        test_losses = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config.device), target.to(config.device)
                output = model(data)
                loss = F.cross_entropy(output, target)
                test_losses.append(loss.item())
                
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)
        
        final_accuracy = test_correct / test_total
        final_loss = sum(test_losses) / len(test_losses)
        
        # Calculate energy efficiency metrics
        avg_energy_efficiency = sum(energy_efficiency) / len(energy_efficiency) if energy_efficiency else 0
        
        # Calculate phase usage statistics
        if phase_distributions:
            avg_phase_dist = {}
            for phase in ['active', 'sleep', 'interactive', 'inspired']:
                avg_phase_dist[f'avg_{phase}_ratio'] = sum(d[phase] for d in phase_distributions) / len(phase_distributions)
        else:
            avg_phase_dist = {}
        
        # Store results for this configuration
        results[phase_config['name']] = {
            'disabled_phases': disabled_phases,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'training_metrics': training_metrics,
            'energy_efficiency': avg_energy_efficiency,
            'phase_distribution': avg_phase_dist,
            'convergence_speed': len([m for m in training_metrics if m['loss'] > final_loss * 1.1])
        }
    
    # Calculate comparative metrics
    baseline_acc = results['full']['final_accuracy']
    
    for config_name, result in results.items():
        if config_name != 'full':
            result['accuracy_drop'] = baseline_acc - result['final_accuracy']
            result['relative_performance'] = result['final_accuracy'] / baseline_acc
    
    return {
        'configurations': results,
        'baseline_accuracy': baseline_acc,
        'summary': {
            'best_config': max(results.keys(), key=lambda k: results[k]['final_accuracy']),
            'worst_config': min(results.keys(), key=lambda k: results[k]['final_accuracy']),
            'most_efficient': max(results.keys(), key=lambda k: results[k]['energy_efficiency']),
        }
    }


def anxiety_restorative_analysis(
    model: AdaptiveModel, stress_conditions: dict[str, Any]
) -> dict[str, Any]:
    """
    Anxiety and restorative behavior analysis.

    This function analyzes how the network responds to
    stress conditions and recovers through restorative mechanisms.

    Args:
        model: Adaptive neural network model
        stress_conditions: Dictionary defining stress scenarios

    Returns:
        Results analyzing stress response and recovery
    """
    device = torch.device(model.config.device)
    results = {}
    
    # Default stress conditions if not provided
    default_stress = {
        'high_loss_threshold': 2.0,
        'conflicting_signals_prob': 0.3,
        'noise_level': 0.5,
        'stress_duration': 20,  # number of steps
        'recovery_duration': 30  # number of steps to observe recovery
    }
    stress_conditions = {**default_stress, **stress_conditions}
    
    # Create stress scenario datasets
    def create_stress_scenario(scenario_type: str, batch_size: int = 32):
        """Create different stress scenarios."""
        input_dim = model.config.input_dim
        
        if scenario_type == 'high_loss':
            # Create data that should produce high loss (random labels)
            data = torch.randn(batch_size, input_dim, device=device)
            labels = torch.randint(0, model.config.output_dim, (batch_size,), device=device)
            return data, labels
            
        elif scenario_type == 'conflicting_signals':
            # Create data with conflicting patterns
            data = torch.randn(batch_size, input_dim, device=device)
            # Add conflicting patterns to same input
            data[:, :input_dim//2] += 2.0  # Pattern A
            data[:, input_dim//2:] -= 2.0   # Conflicting pattern B  
            labels = torch.randint(0, model.config.output_dim, (batch_size,), device=device)
            return data, labels
            
        elif scenario_type == 'noisy_input':
            # Create clean data with added noise
            data = torch.randn(batch_size, input_dim, device=device)
            noise = torch.randn_like(data) * stress_conditions['noise_level']
            data += noise
            labels = torch.randint(0, model.config.output_dim, (batch_size,), device=device)
            return data, labels
            
        else:  # normal
            data = torch.randn(batch_size, input_dim, device=device)
            labels = torch.randint(0, model.config.output_dim, (batch_size,), device=device)
            return data, labels
    
    # Test different stress scenarios
    scenarios = ['normal', 'high_loss', 'conflicting_signals', 'noisy_input']
    
    for scenario in scenarios:
        print(f"\n=== Testing scenario: {scenario} ===")
        
        # Reset model state for each scenario
        model.reset_state()
        scenario_results = {
            'anxiety_progression': [],
            'energy_progression': [],
            'phase_progression': [],
            'restorative_metrics': [],
            'recovery_metrics': []
        }
        
        # Baseline phase (normal operation)
        baseline_steps = 10
        for step in range(baseline_steps):
            data, labels = create_stress_scenario('normal')
            
            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output, labels)
            
            # Track metrics
            if hasattr(model.phase_scheduler, 'node_anxiety'):
                anxiety_stats = model.phase_scheduler.get_anxiety_stats()
                scenario_results['anxiety_progression'].append({
                    'step': step,
                    'phase': 'baseline',
                    **anxiety_stats
                })
            
            if hasattr(model.node_state, 'energy'):
                energy_mean = model.node_state.energy.mean().item()
                scenario_results['energy_progression'].append({
                    'step': step,
                    'phase': 'baseline', 
                    'mean_energy': energy_mean
                })
            
            # Track phase distribution
            phase_stats = model.phase_scheduler.get_phase_stats(
                model.phase_scheduler.node_phases.unsqueeze(0)
            )
            scenario_results['phase_progression'].append({
                'step': step,
                'phase': 'baseline',
                **phase_stats
            })
        
        # Stress phase
        stress_duration = stress_conditions['stress_duration']
        for step in range(stress_duration):
            data, labels = create_stress_scenario(scenario)
            
            # Apply stress through forward pass
            output = model(data)
            loss = F.cross_entropy(output, labels)
            
            # Simulate anxiety increase due to high loss
            if scenario != 'normal' and hasattr(model.phase_scheduler, 'node_anxiety'):
                # Increase anxiety based on loss
                loss_factor = min(2.0, loss.item() / stress_conditions['high_loss_threshold'])
                anxiety_increase = torch.full_like(
                    model.phase_scheduler.node_anxiety, 
                    loss_factor * 0.5
                )
                model.phase_scheduler.node_anxiety += anxiety_increase
                model.phase_scheduler.node_anxiety = torch.clamp(
                    model.phase_scheduler.node_anxiety, 0.0, 15.0
                )
            
            # Track metrics during stress
            if hasattr(model.phase_scheduler, 'node_anxiety'):
                anxiety_stats = model.phase_scheduler.get_anxiety_stats()
                scenario_results['anxiety_progression'].append({
                    'step': baseline_steps + step,
                    'phase': 'stress',
                    **anxiety_stats
                })
            
            if hasattr(model.node_state, 'energy'):
                energy_mean = model.node_state.energy.mean().item()
                scenario_results['energy_progression'].append({
                    'step': baseline_steps + step,
                    'phase': 'stress',
                    'mean_energy': energy_mean
                })
                
            phase_stats = model.phase_scheduler.get_phase_stats(
                model.phase_scheduler.node_phases.unsqueeze(0)
            )
            scenario_results['phase_progression'].append({
                'step': baseline_steps + step,
                'phase': 'stress',
                **phase_stats
            })
        
        # Recovery phase
        recovery_duration = stress_conditions['recovery_duration']
        for step in range(recovery_duration):
            data, labels = create_stress_scenario('normal')
            
            with torch.no_grad():
                output = model(data)
                loss = F.cross_entropy(output, labels)
            
            # Track recovery metrics
            if hasattr(model.phase_scheduler, 'node_anxiety'):
                anxiety_stats = model.phase_scheduler.get_anxiety_stats()
                scenario_results['anxiety_progression'].append({
                    'step': baseline_steps + stress_duration + step,
                    'phase': 'recovery',
                    **anxiety_stats
                })
                
                # Calculate recovery metrics
                if step > 0:  # After first recovery step
                    prev_anxiety = scenario_results['anxiety_progression'][-2]['mean_anxiety']
                    curr_anxiety = anxiety_stats['mean_anxiety']
                    recovery_rate = max(0, prev_anxiety - curr_anxiety) / (prev_anxiety + 1e-8)
                    
                    scenario_results['recovery_metrics'].append({
                        'step': step,
                        'recovery_rate': recovery_rate,
                        'anxiety_reduction': max(0, prev_anxiety - curr_anxiety)
                    })
            
            if hasattr(model.node_state, 'energy'):
                energy_mean = model.node_state.energy.mean().item()
                scenario_results['energy_progression'].append({
                    'step': baseline_steps + stress_duration + step,
                    'phase': 'recovery',
                    'mean_energy': energy_mean
                })
            
            phase_stats = model.phase_scheduler.get_phase_stats(
                model.phase_scheduler.node_phases.unsqueeze(0)
            )
            scenario_results['phase_progression'].append({
                'step': baseline_steps + stress_duration + step, 
                'phase': 'recovery',
                **phase_stats
            })
        
        # Calculate scenario summary metrics
        if scenario_results['anxiety_progression']:
            stress_anxiety_levels = [
                m['mean_anxiety'] for m in scenario_results['anxiety_progression'] 
                if m['phase'] == 'stress'
            ]
            recovery_anxiety_levels = [
                m['mean_anxiety'] for m in scenario_results['anxiety_progression']
                if m['phase'] == 'recovery'  
            ]
            
            scenario_results['summary'] = {
                'max_stress_anxiety': max(stress_anxiety_levels) if stress_anxiety_levels else 0,
                'final_recovery_anxiety': recovery_anxiety_levels[-1] if recovery_anxiety_levels else 0,
                'anxiety_resilience': 1.0 - (max(stress_anxiety_levels) if stress_anxiety_levels else 0) / 15.0,
                'recovery_effectiveness': len([m for m in scenario_results['recovery_metrics'] if m['recovery_rate'] > 0.1]) / len(scenario_results['recovery_metrics']) if scenario_results['recovery_metrics'] else 0
            }
        
        results[scenario] = scenario_results
    
    # Overall analysis
    results['overall_analysis'] = {
        'most_stressful_scenario': max(
            scenarios, 
            key=lambda s: results[s].get('summary', {}).get('max_stress_anxiety', 0)
        ),
        'best_recovery_scenario': max(
            scenarios,
            key=lambda s: results[s].get('summary', {}).get('recovery_effectiveness', 0)
        ),
        'stress_sensitivity': sum(
            results[s].get('summary', {}).get('max_stress_anxiety', 0) 
            for s in scenarios if s != 'normal'
        ) / (len(scenarios) - 1),
        'average_resilience': sum(
            results[s].get('summary', {}).get('anxiety_resilience', 0)
            for s in scenarios
        ) / len(scenarios)
    }
    
    return results
