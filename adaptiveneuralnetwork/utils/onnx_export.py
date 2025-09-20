"""
ONNX export and model introspection utilities for adaptive neural networks.

This module provides functionality to export models to ONNX format and
perform comprehensive model introspection for debugging and optimization.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..api.model import AdaptiveModel


class ModelIntrospection:
    """Comprehensive model introspection utilities."""
    
    def __init__(self, model: AdaptiveModel):
        self.model = model
        self.config = model.config
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary including parameters, memory, and structure."""
        summary = {
            "architecture": {
                "num_nodes": self.config.num_nodes,
                "hidden_dim": self.config.hidden_dim,
                "input_dim": self.config.input_dim,
                "output_dim": self.config.output_dim,
            },
            "parameters": self._get_parameter_info(),
            "memory": self._get_memory_info(),
            "structure": self._get_structure_info(),
            "device": str(self.model.input_projection.weight.device),
        }
        return summary
    
    def _get_parameter_info(self) -> Dict[str, Any]:
        """Get detailed parameter information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        param_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "parameters_by_layer": {},
        }
        
        for name, param in self.model.named_parameters():
            param_info["parameters_by_layer"][name] = {
                "shape": list(param.shape),
                "numel": param.numel(),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
            }
        
        return param_info
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        memory_info = {
            "estimated_size_mb": self._estimate_model_size_mb(),
            "state_tensors": {},
        }
        
        # Add node state memory info
        if hasattr(self.model, 'node_state'):
            memory_info["state_tensors"]["node_energy"] = {
                "shape": list(self.model.node_state.energy.shape),
                "dtype": str(self.model.node_state.energy.dtype),
                "size_kb": self.model.node_state.energy.numel() * self.model.node_state.energy.element_size() / 1024
            }
            memory_info["state_tensors"]["node_activity"] = {
                "shape": list(self.model.node_state.activity.shape),
                "dtype": str(self.model.node_state.activity.dtype),
                "size_kb": self.model.node_state.activity.numel() * self.model.node_state.activity.element_size() / 1024
            }
        
        return memory_info
    
    def _estimate_model_size_mb(self) -> float:
        """Estimate model size in MB."""
        total_bytes = 0
        for param in self.model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes / (1024 * 1024)
    
    def _get_structure_info(self) -> Dict[str, Any]:
        """Get model structure information."""
        structure = {
            "modules": [],
            "activation_functions": [],
            "layers": {},
        }
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module_info = {
                    "name": name,
                    "type": type(module).__name__,
                    "parameters": sum(p.numel() for p in module.parameters()),
                }
                structure["modules"].append(module_info)
                
                # Track layer types
                layer_type = type(module).__name__
                if layer_type not in structure["layers"]:
                    structure["layers"][layer_type] = 0
                structure["layers"][layer_type] += 1
        
        return structure
    
    def analyze_gradient_flow(self, loss: torch.Tensor) -> Dict[str, Any]:
        """Analyze gradient flow through the model."""
        gradient_info = {
            "has_gradients": {},
            "gradient_norms": {},
            "zero_gradients": [],
        }
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_info["has_gradients"][name] = True
                gradient_info["gradient_norms"][name] = grad_norm
                
                if grad_norm < 1e-8:
                    gradient_info["zero_gradients"].append(name)
            else:
                gradient_info["has_gradients"][name] = False
        
        return gradient_info


class ONNXExporter:
    """ONNX export utilities for adaptive neural networks."""
    
    def __init__(self, model: AdaptiveModel):
        self.model = model
        self.config = model.config
    
    def export_to_onnx(
        self,
        filepath: Union[str, Path],
        input_shape: Optional[Tuple[int, ...]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        opset_version: int = 11,
        **kwargs
    ) -> bool:
        """
        Export the model to ONNX format.
        
        Args:
            filepath: Path to save the ONNX model
            input_shape: Input tensor shape (batch_size, input_dim)
            dynamic_axes: Dynamic axes for variable input sizes
            opset_version: ONNX opset version
            **kwargs: Additional arguments for torch.onnx.export
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            # Default input shape if not provided
            if input_shape is None:
                input_shape = (1, self.config.input_dim)
            
            # Create dummy input
            dummy_input = torch.randn(input_shape, device=self.config.device)
            
            # Default dynamic axes for batch size flexibility
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Set model to eval mode
            self.model.eval()
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                str(filepath),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                **kwargs
            )
            
            return True
            
        except Exception as e:
            warnings.warn(f"ONNX export failed: {e}", stacklevel=2)
            return False
    
    def verify_onnx_export(
        self,
        onnx_filepath: Union[str, Path],
        test_input: Optional[torch.Tensor] = None,
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        Verify ONNX export by comparing outputs.
        
        Args:
            onnx_filepath: Path to the exported ONNX model
            test_input: Test input tensor (if None, creates random input)
            tolerance: Tolerance for output comparison
            
        Returns:
            Verification results dictionary
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            return {
                "success": False,
                "error": "ONNX or ONNXRuntime not installed. Install with: pip install onnx onnxruntime"
            }
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(str(onnx_filepath))
            onnx.checker.check_model(onnx_model)
            
            # Create ONNX runtime session
            ort_session = ort.InferenceSession(str(onnx_filepath))
            
            # Create test input if not provided
            if test_input is None:
                test_input = torch.randn(1, self.config.input_dim, device=self.config.device)
            
            # Get PyTorch output
            self.model.eval()
            with torch.no_grad():
                pytorch_output = self.model(test_input)
            
            # Get ONNX output
            ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
            onnx_output = ort_session.run(None, ort_inputs)[0]
            
            # Compare outputs
            pytorch_output_np = pytorch_output.cpu().numpy()
            max_diff = abs(pytorch_output_np - onnx_output).max()
            
            verification_result = {
                "success": True,
                "max_difference": float(max_diff),
                "within_tolerance": max_diff < tolerance,
                "pytorch_output_shape": pytorch_output_np.shape,
                "onnx_output_shape": onnx_output.shape,
                "model_valid": True,
            }
            
            return verification_result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


def export_model_with_introspection(
    model: AdaptiveModel,
    output_dir: Union[str, Path],
    export_onnx: bool = True,
    create_summary: bool = True,
) -> Dict[str, Any]:
    """
    Export model with full introspection report.
    
    Args:
        model: The adaptive model to export
        output_dir: Directory to save outputs
        export_onnx: Whether to export ONNX format
        create_summary: Whether to create introspection summary
        
    Returns:
        Dictionary with export results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "output_directory": str(output_dir),
        "files_created": [],
        "errors": [],
    }
    
    # Model introspection
    if create_summary:
        try:
            introspector = ModelIntrospection(model)
            summary = introspector.get_model_summary()
            
            # Save summary as JSON
            import json
            summary_path = output_dir / "model_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            results["files_created"].append(str(summary_path))
            results["model_summary"] = summary
            
        except Exception as e:
            results["errors"].append(f"Introspection failed: {e}")
    
    # ONNX export
    if export_onnx:
        try:
            exporter = ONNXExporter(model)
            onnx_path = output_dir / "model.onnx"
            
            if exporter.export_to_onnx(onnx_path):
                results["files_created"].append(str(onnx_path))
                
                # Verify export
                verification = exporter.verify_onnx_export(onnx_path)
                results["onnx_verification"] = verification
                
                # Save verification report
                import json
                verify_path = output_dir / "onnx_verification.json"
                with open(verify_path, 'w') as f:
                    json.dump(verification, f, indent=2)
                results["files_created"].append(str(verify_path))
                
            else:
                results["errors"].append("ONNX export failed")
                
        except Exception as e:
            results["errors"].append(f"ONNX export error: {e}")
    
    return results
