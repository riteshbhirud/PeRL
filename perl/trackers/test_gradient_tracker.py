# Copyright (c) 2025
# Test script for GradientFlowTracker
# Run with: python -m perl.trackers.test_gradient_tracker

import torch
import torch.nn as nn
import tempfile
import os
import numpy as np

from .gradient_tracker import GradientFlowTracker


class MockLoRALayer(nn.Module):
    """Mock LoRA layer for testing."""

    def __init__(self, in_features=64, out_features=64, rank=8):
        super().__init__()
        self.lora_A = nn.ModuleDict({
            'default': nn.Linear(in_features, rank, bias=False)
        })
        self.lora_B = nn.ModuleDict({
            'default': nn.Linear(rank, out_features, bias=False)
        })
        # Initialize both weights with small values so gradients flow
        nn.init.normal_(self.lora_A['default'].weight, std=0.02)
        nn.init.normal_(self.lora_B['default'].weight, std=0.02)

    def forward(self, x):
        # Actually use the LoRA parameters to create gradients
        delta = self.lora_B['default'](self.lora_A['default'](x))
        return x + delta


class MockDoRALayer(nn.Module):
    """Mock DoRA layer with magnitude vector for testing."""

    def __init__(self, in_features=64, out_features=64, rank=8):
        super().__init__()
        self.lora_A = nn.ModuleDict({
            'default': nn.Linear(in_features, rank, bias=False)
        })
        self.lora_B = nn.ModuleDict({
            'default': nn.Linear(rank, out_features, bias=False)
        })
        self.lora_magnitude_vector = nn.Parameter(torch.ones(out_features))
        # Initialize both weights with small values so gradients flow
        nn.init.normal_(self.lora_A['default'].weight, std=0.02)
        nn.init.normal_(self.lora_B['default'].weight, std=0.02)

    def forward(self, x):
        # Use LoRA parameters with magnitude scaling
        delta = self.lora_B['default'](self.lora_A['default'](x))
        # Apply magnitude vector (simplified)
        return x + delta * self.lora_magnitude_vector.unsqueeze(0)


class MockSelfAttn(nn.Module):
    """Mock self-attention module."""

    def __init__(self, layer_class=MockLoRALayer):
        super().__init__()
        self.q_proj = layer_class()
        self.k_proj = layer_class()
        self.v_proj = layer_class()
        self.o_proj = layer_class()

    def forward(self, x):
        # Use all projections to ensure gradients flow
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = self.o_proj(q + k + v)
        return out


class MockMLP(nn.Module):
    """Mock MLP module."""

    def __init__(self, layer_class=MockLoRALayer):
        super().__init__()
        self.up_proj = layer_class()
        self.down_proj = layer_class()
        self.gate_proj = layer_class()

    def forward(self, x):
        # Use all projections to ensure gradients flow
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        out = self.down_proj(up * gate)
        return out


class MockTransformerLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, layer_class=MockLoRALayer):
        super().__init__()
        self.self_attn = MockSelfAttn(layer_class)
        self.mlp = MockMLP(layer_class)

    def forward(self, x):
        # Pass through both attn and mlp
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockModel(nn.Module):
    """Mock transformer model with nested structure."""

    def __init__(self, num_layers=4, layer_class=MockLoRALayer):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer(layer_class) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def run_test(test_name: str, test_fn):
    """Run a single test with error handling."""
    try:
        test_fn()
        print(f"  ✓ {test_name}")
        return True
    except Exception as e:
        print(f"  ✗ {test_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_tracking():
    """Test basic gradient flow tracking."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Verify adapter parameters were discovered
        assert len(tracker.adapter_params) > 0, "Should discover adapter parameters"

        # Simulate training steps with gradients
        for step in range(5):
            # Create gradients by doing forward/backward
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()

            # Log gradients
            logged = tracker.log_step(step)
            assert logged, f"Should log at step {step}"

            # Zero gradients for next iteration
            model.zero_grad()

        # Verify history was recorded
        assert len(tracker.steps_logged) == 5, f"Should have 5 steps, got {len(tracker.steps_logged)}"
        assert len(tracker.gradient_flow) == 5, "Should have 5 entries in gradient_flow"
        assert len(tracker.layer_flow) == 5, "Should have 5 entries in layer_flow"


def test_layer_extraction():
    """Test layer index and component extraction."""
    model = MockModel(num_layers=8)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Test layer extraction from parameter names
        test_names = [
            ("model.layers.0.self_attn.q_proj.lora_A.default.weight", 0),
            ("model.layers.5.mlp.up_proj.lora_B.default.weight", 5),
            ("base_model.model.model.layers.15.self_attn.v_proj.lora_A.weight", 15),
        ]

        for name, expected_idx in test_names:
            idx = tracker._extract_layer_idx(name)
            assert idx == expected_idx, f"Expected layer {expected_idx} from {name}, got {idx}"

        # Test component extraction
        component_tests = [
            ("lora_A.default.weight", "A"),
            ("lora_B.weight", "B"),
            ("lora_magnitude_vector", "magnitude"),
            ("lora_E", "E"),
        ]

        for name, expected_comp in component_tests:
            comp = tracker._extract_component(name)
            assert comp == expected_comp, f"Expected component {expected_comp} from {name}, got {comp}"


def test_gradient_matrix():
    """Test gradient matrix generation for heatmaps."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Simulate training with varying gradients
        for step in range(10):
            x = torch.randn(2, 64, requires_grad=True) * (step + 1)
            output = model(x)
            loss = output.sum()
            loss.backward()
            tracker.log_step(step)
            model.zero_grad()

        # Get gradient matrix for component A
        matrix, layer_indices, steps = tracker.get_gradient_matrix(component='A')

        assert matrix.ndim == 2, "Matrix should be 2D"
        assert len(layer_indices) == 4, f"Should have 4 layers, got {len(layer_indices)}"
        assert len(steps) == 10, f"Should have 10 steps, got {len(steps)}"
        assert matrix.shape == (4, 10), f"Matrix shape should be (4, 10), got {matrix.shape}"

        # Verify non-zero values exist
        assert np.any(matrix > 0), "Matrix should have non-zero gradient values"


def test_dora_tracking():
    """Test tracking DoRA-specific magnitude gradients."""
    model = MockModel(num_layers=4, layer_class=MockDoRALayer)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='dora',
            verbose=False,
        )

        # Verify magnitude parameters are tracked
        magnitude_params = [n for n in tracker.adapter_params if 'magnitude' in n.lower()]
        assert len(magnitude_params) > 0, "Should discover magnitude parameters for DoRA"

        # Simulate training
        for step in range(3):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            tracker.log_step(step)
            model.zero_grad()

        # Check that 'magnitude' component is tracked
        assert 'magnitude' in tracker.components, "Should track magnitude component for DoRA"


def test_save_load():
    """Test saving and loading gradient flow data."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Generate some data
        for step in range(5):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            tracker.log_step(step)
            model.zero_grad()

        # Save
        save_path = tracker.save()
        assert os.path.exists(save_path), "Save file should exist"

        # Verify summary JSON was also created
        summary_path = save_path.replace('.pt', '_summary.json')
        assert os.path.exists(summary_path), "Summary JSON should exist"

        # Load
        loaded_tracker = GradientFlowTracker.load(save_path)

        # Verify loaded data matches original
        assert loaded_tracker.peft_type == tracker.peft_type
        assert loaded_tracker.steps_logged == tracker.steps_logged
        assert set(loaded_tracker.layer_indices) == set(tracker.layer_indices)
        assert set(loaded_tracker.components) == set(tracker.components)
        assert len(loaded_tracker.gradient_flow) == len(tracker.gradient_flow)


def test_summary_stats():
    """Test summary statistics generation."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Generate data
        for step in range(5):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            tracker.log_step(step)
            model.zero_grad()

        summary = tracker.get_summary_stats()

        # Check required keys
        assert 'peft_type' in summary
        assert 'total_steps_logged' in summary
        assert 'num_layers' in summary
        assert 'overall_stats' in summary
        assert 'timing' in summary

        # Check overall stats
        overall = summary['overall_stats']
        assert 'mean' in overall
        assert 'std' in overall
        assert 'min' in overall
        assert 'max' in overall

        # Verify values are reasonable
        assert summary['total_steps_logged'] == 5
        assert summary['num_layers'] == 4
        assert overall['max'] >= overall['min']


def test_heatmap_generation():
    """Test heatmap visualization (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⊘ Heatmap generation: matplotlib not available, skipping")
        return

    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Generate data
        for step in range(10):
            x = torch.randn(2, 64, requires_grad=True) * (step + 1)
            output = model(x)
            loss = output.sum()
            loss.backward()
            tracker.log_step(step)
            model.zero_grad()

        # Test heatmap creation
        heatmap_path = os.path.join(tmpdir, 'test_heatmap.png')
        fig = tracker.create_heatmap(
            component='B',
            save_path=heatmap_path,
            log_scale=True,
        )

        assert fig is not None, "Should return figure object"
        assert os.path.exists(heatmap_path), "Heatmap file should be saved"
        plt.close(fig)

        # Test layer comparison plot
        comparison_path = os.path.join(tmpdir, 'test_comparison.png')
        fig = tracker.create_layer_comparison_plot(
            save_path=comparison_path,
        )

        assert fig is not None, "Should return figure object"
        assert os.path.exists(comparison_path), "Comparison plot should be saved"
        plt.close(fig)


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("GradientFlowTracker Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Basic gradient tracking", test_basic_tracking),
        ("Layer and component extraction", test_layer_extraction),
        ("Gradient matrix generation", test_gradient_matrix),
        ("DoRA magnitude tracking", test_dora_tracking),
        ("Save and load", test_save_load),
        ("Summary statistics", test_summary_stats),
        ("Heatmap generation", test_heatmap_generation),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        if run_test(test_name, test_fn):
            passed += 1
        else:
            failed += 1

    print("\n" + "-" * 60)
    print(f"Results: {passed}/{passed + failed} tests passed")

    if failed == 0:
        print("All tests passed!")
    else:
        print(f"Warning: {failed} test(s) failed")

    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
