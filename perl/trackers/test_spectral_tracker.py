#!/usr/bin/env python3
"""
Test script to verify SpectralTracker implementation.

This script tests the SpectralTracker with different PEFT methods
on a small model to ensure correct functionality.

Usage:
    python -m perl.trackers.test_spectral_tracker

    # Or with specific PEFT type:
    python -m perl.trackers.test_spectral_tracker --peft_type dora
"""

import os
import sys
import tempfile
import argparse
import torch
import numpy as np
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from perl.trackers.spectral_tracker import SpectralTracker, SpectralMetrics


def create_mock_lora_model(peft_type: str = "lora", rank: int = 8, hidden_size: int = 256):
    """
    Create a mock model with LoRA-style adapters for testing.

    This avoids loading a large LLM for quick testing.
    """
    import torch.nn as nn

    class MockLoRALayer(nn.Module):
        """Mock LoRA layer that mimics PEFT library structure."""
        def __init__(self, in_features, out_features, r, use_dora=False):
            super().__init__()
            self.r = r
            self.use_dora = use_dora

            # Base layer
            self.base_layer = nn.Linear(in_features, out_features, bias=False)

            # LoRA adapters (dict-style like PEFT)
            self.lora_A = nn.ModuleDict({
                'default': nn.Linear(in_features, r, bias=False)
            })
            self.lora_B = nn.ModuleDict({
                'default': nn.Linear(r, out_features, bias=False)
            })

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A['default'].weight)
            nn.init.zeros_(self.lora_B['default'].weight)

            # DoRA magnitude vector
            if use_dora:
                self.lora_magnitude_vector = nn.ParameterDict({
                    'default': nn.Parameter(torch.ones(out_features))
                })

        def forward(self, x):
            base_out = self.base_layer(x)
            lora_out = self.lora_B['default'](self.lora_A['default'](x))
            return base_out + lora_out

    class MockAdaLoRALayer(nn.Module):
        """Mock AdaLoRA layer with importance scores."""
        def __init__(self, in_features, out_features, r):
            super().__init__()
            self.r = r

            self.base_layer = nn.Linear(in_features, out_features, bias=False)

            self.lora_A = nn.ModuleDict({
                'default': nn.Linear(in_features, r, bias=False)
            })
            self.lora_B = nn.ModuleDict({
                'default': nn.Linear(r, out_features, bias=False)
            })
            self.lora_E = nn.ParameterDict({
                'default': nn.Parameter(torch.ones(r))  # Importance scores
            })

            nn.init.kaiming_uniform_(self.lora_A['default'].weight)
            nn.init.zeros_(self.lora_B['default'].weight)

        def forward(self, x):
            base_out = self.base_layer(x)
            # AdaLoRA: B @ diag(E) @ A
            A_out = self.lora_A['default'](x)
            scaled = A_out * self.lora_E['default']
            lora_out = self.lora_B['default'](scaled)
            return base_out + lora_out

    class MockVeRALayer(nn.Module):
        """Mock VeRA layer with frozen random matrices."""
        def __init__(self, in_features, out_features, r):
            super().__init__()
            self.r = r

            self.base_layer = nn.Linear(in_features, out_features, bias=False)

            # Frozen random matrices (registered as buffers but also accessible as attributes)
            self.vera_A = torch.randn(r, in_features) * 0.01
            self.vera_B = torch.randn(out_features, r) * 0.01
            self.register_buffer('_vera_A', self.vera_A)
            self.register_buffer('_vera_B', self.vera_B)

            # Trainable scaling vectors
            self.vera_lambda_b = nn.Parameter(torch.ones(out_features))
            self.vera_lambda_d = nn.Parameter(torch.ones(r))

        def forward(self, x):
            base_out = self.base_layer(x)
            # VeRA: diag(λ_b) @ B @ diag(λ_d) @ A @ x
            A_out = x @ self.vera_A.T  # (batch, r)
            scaled_A = A_out * self.vera_lambda_d
            B_out = scaled_A @ self.vera_B.T  # (batch, out)
            lora_out = B_out * self.vera_lambda_b
            return base_out + lora_out

    class MockSelfAttn(nn.Module):
        """Mock self-attention module with q_proj adapter."""
        def __init__(self, hidden_size, r, peft_type):
            super().__init__()
            if peft_type == "lora":
                self.q_proj = MockLoRALayer(hidden_size, hidden_size, r)
            elif peft_type == "dora":
                self.q_proj = MockLoRALayer(hidden_size, hidden_size, r, use_dora=True)
            elif peft_type == "adalora":
                self.q_proj = MockAdaLoRALayer(hidden_size, hidden_size, r)
            elif peft_type == "vera":
                self.q_proj = MockVeRALayer(hidden_size, hidden_size, r)
            else:
                self.q_proj = MockLoRALayer(hidden_size, hidden_size, r)

        def forward(self, x):
            return self.q_proj(x)

    class MockTransformerLayer(nn.Module):
        """Mock transformer layer."""
        def __init__(self, hidden_size, r, peft_type):
            super().__init__()
            self.self_attn = MockSelfAttn(hidden_size, r, peft_type)

        def forward(self, x):
            return self.self_attn(x)

    class MockModel(nn.Module):
        """Mock model inner structure."""
        def __init__(self, num_layers, hidden_size, r, peft_type):
            super().__init__()
            self.layers = nn.ModuleList([
                MockTransformerLayer(hidden_size, r, peft_type)
                for _ in range(num_layers)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    class MockPEFTModel(nn.Module):
        """Mock model with multiple PEFT layers (mimics real model structure)."""
        def __init__(self, num_layers=4, hidden_size=256, r=8, peft_type="lora"):
            super().__init__()
            # Create nested structure: model.layers.N.self_attn.q_proj
            self.model = MockModel(num_layers, hidden_size, r, peft_type)

        def forward(self, x):
            return self.model(x)

    return MockPEFTModel(num_layers=4, hidden_size=hidden_size, r=rank, peft_type=peft_type)


def test_spectral_tracker_basic():
    """Test basic SpectralTracker functionality."""
    print("\n" + "="*60)
    print("TEST: Basic SpectralTracker functionality")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock model
        model = create_mock_lora_model(peft_type="lora", rank=8)

        # Create tracker
        tracker = SpectralTracker(
            model=model,
            log_frequency=1,  # Log every step for testing
            save_dir=tmpdir,
            peft_type="lora",
            verbose=True,
        )

        # Check adapter discovery
        print(f"\nDiscovered {len(tracker.adapter_layers)} adapter layers:")
        for name in tracker.adapter_layers:
            print(f"  - {name}")

        assert len(tracker.adapter_layers) > 0, "No adapter layers discovered!"

        # Simulate training steps
        for step in range(1, 6):
            # Modify weights slightly to simulate training
            for name, module in model.named_modules():
                if hasattr(module, 'lora_B'):
                    with torch.no_grad():
                        module.lora_B['default'].weight.data += torch.randn_like(
                            module.lora_B['default'].weight
                        ) * 0.1

            # Log step
            logged = tracker.log_step(step)
            print(f"Step {step}: logged={logged}")

        # Check history
        assert len(tracker.steps_logged) == 5, f"Expected 5 steps logged, got {len(tracker.steps_logged)}"

        # Get summary
        summary = tracker.get_summary_stats()
        print(f"\nSummary stats:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        # Save and reload
        save_path = tracker.save()
        print(f"\nSaved to: {save_path}")

        # Load and verify
        loaded_tracker = SpectralTracker.load(save_path)
        assert len(loaded_tracker.steps_logged) == 5
        print("Loaded tracker successfully!")

        print("\n✓ Basic test PASSED")
        return True


def test_spectral_metrics():
    """Test spectral metric computations."""
    print("\n" + "="*60)
    print("TEST: Spectral metric computations")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_mock_lora_model(peft_type="lora", rank=8)

        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            verbose=False,
        )

        # Log a step
        tracker.log_step(1)

        # Check metrics for first layer
        layer_name = list(tracker.adapter_layers.keys())[0]
        history = tracker.get_layer_history(layer_name)

        print(f"\nMetrics for {layer_name}:")

        # Check all expected metrics exist
        expected_metrics = [
            'singular_values', 'spectral_gap', 'condition_number',
            'effective_rank', 'nuclear_norm', 'frobenius_norm',
            'top1_sv_ratio', 'top5_sv_ratio', 'top10_sv_ratio',
            'matrix_shape', 'rank', 'steps'
        ]

        for metric in expected_metrics:
            assert metric in history, f"Missing metric: {metric}"
            value = history[metric][0]
            print(f"  {metric}: {value}")

        # Validate metric ranges (note: mock B is initialized to zeros, so values may be degenerate)
        assert history['effective_rank'][0] >= 0, "Effective rank negative"
        assert history['condition_number'][0] >= 1, "Condition number < 1"  # Should be 1.0 for zero matrix
        assert 0 <= history['top1_sv_ratio'][0] <= 1, "Top-1 ratio out of range"

        print("\n✓ Metrics test PASSED")
        return True


def test_dora_specific():
    """Test DoRA-specific metric tracking."""
    print("\n" + "="*60)
    print("TEST: DoRA-specific metrics")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_mock_lora_model(peft_type="dora", rank=8)

        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type="dora",
            verbose=False,
        )

        assert tracker.peft_type == "dora", "PEFT type not detected as dora"

        # Log a step
        tracker.log_step(1)

        # Check DoRA-specific metrics
        layer_name = list(tracker.adapter_layers.keys())[0]
        history = tracker.get_layer_history(layer_name)

        print(f"\nDoRA metrics for {layer_name}:")

        if 'magnitude_norm' in history:
            print(f"  magnitude_norm: {history['magnitude_norm'][0]}")
        if 'direction_norm' in history:
            print(f"  direction_norm: {history['direction_norm'][0]}")
        if 'magnitude_direction_ratio' in history:
            print(f"  magnitude_direction_ratio: {history['magnitude_direction_ratio'][0]}")

        # DoRA should have magnitude tracking
        assert 'magnitude_norm' in history, "DoRA magnitude_norm not tracked"

        print("\n✓ DoRA test PASSED")
        return True


def test_adalora_specific():
    """Test AdaLoRA-specific metric tracking."""
    print("\n" + "="*60)
    print("TEST: AdaLoRA-specific metrics")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_mock_lora_model(peft_type="adalora", rank=8)

        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type="adalora",
            verbose=False,
        )

        assert tracker.peft_type == "adalora", "PEFT type not detected as adalora"

        # Log a step
        tracker.log_step(1)

        # Check AdaLoRA-specific metrics
        layer_name = list(tracker.adapter_layers.keys())[0]
        history = tracker.get_layer_history(layer_name)

        print(f"\nAdaLoRA metrics for {layer_name}:")

        if 'importance_scores' in history:
            scores = history['importance_scores'][0]
            print(f"  importance_scores shape: {np.array(scores).shape}")
            print(f"  importance_scores: {scores}")

        print("\n✓ AdaLoRA test PASSED")
        return True


def test_vera_specific():
    """Test VeRA-specific metric tracking."""
    print("\n" + "="*60)
    print("TEST: VeRA-specific metrics")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_mock_lora_model(peft_type="vera", rank=8)

        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type="vera",
            verbose=False,
        )

        assert tracker.peft_type == "vera", "PEFT type not detected as vera"

        # Log a step
        tracker.log_step(1)

        # VeRA should still produce spectral metrics
        layer_name = list(tracker.adapter_layers.keys())[0]
        history = tracker.get_layer_history(layer_name)

        print(f"\nVeRA metrics for {layer_name}:")
        print(f"  singular_values shape: {np.array(history['singular_values'][0]).shape}")
        print(f"  effective_rank: {history['effective_rank'][0]}")

        print("\n✓ VeRA test PASSED")
        return True


def test_performance():
    """Test that spectral tracking has acceptable overhead."""
    print("\n" + "="*60)
    print("TEST: Performance overhead")
    print("="*60)

    import time

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create larger model for timing
        model = create_mock_lora_model(peft_type="lora", rank=32, hidden_size=512)

        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            verbose=False,
        )

        # Time logging
        num_iterations = 10
        times = []

        for i in range(num_iterations):
            start = time.time()
            tracker.log_step(i + 1)
            elapsed = time.time() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        max_time = np.max(times)

        print(f"\nTiming for {len(tracker.adapter_layers)} layers:")
        print(f"  Average: {avg_time*1000:.2f} ms")
        print(f"  Max: {max_time*1000:.2f} ms")
        print(f"  Total for {num_iterations} logs: {sum(times):.2f} s")

        # Check overhead is reasonable (< 1s per log for mock model)
        assert avg_time < 1.0, f"Average logging time too high: {avg_time}s"

        print("\n✓ Performance test PASSED")
        return True


def test_save_load():
    """Test saving and loading spectral history."""
    print("\n" + "="*60)
    print("TEST: Save and load functionality")
    print("="*60)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = create_mock_lora_model(peft_type="lora", rank=8)

        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            verbose=False,
        )

        # Log several steps
        for step in [100, 200, 300]:
            tracker.log_step(step)

        # Save
        save_path = tracker.save()
        summary_path = save_path.replace('.pt', '_summary.json')

        # Check files exist
        assert os.path.exists(save_path), "Main save file not created"
        assert os.path.exists(summary_path), "Summary file not created"

        print(f"\nSaved files:")
        print(f"  {save_path} ({os.path.getsize(save_path)} bytes)")
        print(f"  {summary_path} ({os.path.getsize(summary_path)} bytes)")

        # Load and verify
        loaded = SpectralTracker.load(save_path)

        assert loaded.steps_logged == [100, 200, 300], "Steps mismatch after load"
        assert loaded.peft_type == "lora", "PEFT type mismatch after load"
        assert len(loaded.adapter_layers) == len(tracker.adapter_layers), "Layer count mismatch"

        # Verify history is preserved
        for layer_name in tracker.adapter_layers:
            original = tracker.get_layer_history(layer_name)
            loaded_hist = loaded.get_layer_history(layer_name)

            for metric in original:
                assert metric in loaded_hist, f"Metric {metric} missing after load"
                assert len(original[metric]) == len(loaded_hist[metric]), \
                    f"Metric {metric} length mismatch"

        print("\n✓ Save/load test PASSED")
        return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# SpectralTracker Test Suite")
    print("#"*60)

    tests = [
        ("Basic functionality", test_spectral_tracker_basic),
        ("Spectral metrics", test_spectral_metrics),
        ("DoRA-specific", test_dora_specific),
        ("AdaLoRA-specific", test_adalora_specific),
        ("VeRA-specific", test_vera_specific),
        ("Performance", test_performance),
        ("Save/Load", test_save_load),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, success, error in results:
        status = "✓ PASSED" if success else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")

    print(f"\n{passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SpectralTracker")
    parser.add_argument("--peft_type", type=str, default=None,
                       help="Test specific PEFT type (lora, dora, adalora, vera)")
    args = parser.parse_args()

    if args.peft_type:
        # Run specific test
        print(f"\nTesting PEFT type: {args.peft_type}")

        with tempfile.TemporaryDirectory() as tmpdir:
            model = create_mock_lora_model(peft_type=args.peft_type)
            tracker = SpectralTracker(
                model=model,
                log_frequency=1,
                save_dir=tmpdir,
                peft_type=args.peft_type,
            )

            for step in range(1, 4):
                tracker.log_step(step)

            summary = tracker.get_summary_stats()
            print(f"\nSummary: {summary}")
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
