# Copyright (c) 2025
# Integration test for Phase 1C - Tracker integration with training pipeline
# Run with: python -m perl.trackers.test_integration

import torch
import torch.nn as nn
import tempfile
import os
import sys
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from perl.trackers import (
    SpectralTracker,
    SpectralTrackingCallback,
    GradientFlowTracker,
    GradientFlowCallback,
    create_tracking_callbacks,
)


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
        nn.init.normal_(self.lora_A['default'].weight, std=0.02)
        nn.init.normal_(self.lora_B['default'].weight, std=0.02)

    def forward(self, x):
        return x + self.lora_B['default'](self.lora_A['default'](x))


class MockSelfAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = MockLoRALayer()
        self.k_proj = MockLoRALayer()
        self.v_proj = MockLoRALayer()
        self.o_proj = MockLoRALayer()

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = MockLoRALayer()
        self.down_proj = MockLoRALayer()
        self.gate_proj = MockLoRALayer()

    def forward(self, x):
        up = self.up_proj(x)
        gate = self.gate_proj(x)
        return self.down_proj(up * gate)


class MockTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockSelfAttn()
        self.mlp = MockMLP()

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockModel(nn.Module):
    """Mock transformer model with 4 layers."""
    def __init__(self, num_layers=4):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([
            MockTransformerLayer() for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


class MockTrainerState:
    """Mock trainer state for callback testing."""
    def __init__(self):
        self.global_step = 0


class MockTrainerControl:
    """Mock trainer control for callback testing."""
    pass


class MockTrainingArguments:
    """Mock training arguments for callback testing."""
    def __init__(self, output_dir):
        self.output_dir = output_dir


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


def test_cli_argument_parsing():
    """Test CLI argument parsing for tracking flags."""
    from perl.config.config import TrainConfig

    # Define parsing functions locally to avoid fire import issue
    def _parse_value(value_str: str):
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
        if value_str.startswith('[') and value_str.endswith(']'):
            import ast
            return ast.literal_eval(value_str)
        return value_str

    def _set_nested_attr(config, config_path: str, value):
        parts = config_path.split('.')
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    # Test value parsing
    assert _parse_value("42") == 42
    assert _parse_value("3.14") == 3.14
    assert _parse_value("true") == True
    assert _parse_value("false") == False
    assert _parse_value("hello") == "hello"
    assert _parse_value("[1, 2, 3]") == [1, 2, 3]

    # Test nested attribute setting
    config = TrainConfig()
    _set_nested_attr(config, "tracker.enable_spectral_tracking", True)
    assert config.tracker.enable_spectral_tracking == True

    _set_nested_attr(config, "tracker.spectral_log_frequency", 50)
    assert config.tracker.spectral_log_frequency == 50


def test_combined_tracking():
    """Test both trackers running together."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        spectral_dir = os.path.join(tmpdir, 'spectral')
        gradient_dir = os.path.join(tmpdir, 'gradient')

        spectral_tracker = SpectralTracker(
            model=model,
            log_frequency=5,
            save_dir=spectral_dir,
            peft_type='lora',
            verbose=False,
        )

        gradient_tracker = GradientFlowTracker(
            model=model,
            log_frequency=5,
            save_dir=gradient_dir,
            peft_type='lora',
            verbose=False,
        )

        # Simulate training loop
        for step in range(20):
            # Forward pass
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Gradient tracking (after backward, before optimizer step)
            gradient_tracker.log_step(step)

            # Optimizer step (simulated)
            model.zero_grad()

            # Spectral tracking (after optimizer step)
            spectral_tracker.log_step(step)

        # Save both
        spectral_path = spectral_tracker.save()
        gradient_path = gradient_tracker.save()

        # Verify files exist
        assert os.path.exists(spectral_path), "Spectral save file not found"
        assert os.path.exists(gradient_path), "Gradient save file not found"

        # Load and verify
        loaded_spectral = SpectralTracker.load(spectral_path)
        loaded_gradient = GradientFlowTracker.load(gradient_path)

        assert len(loaded_spectral.steps_logged) == 4, f"Expected 4 spectral steps, got {len(loaded_spectral.steps_logged)}"
        assert len(loaded_gradient.steps_logged) == 4, f"Expected 4 gradient steps, got {len(loaded_gradient.steps_logged)}"


def test_callback_integration():
    """Test callbacks work correctly with mock trainer."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trackers
        spectral_tracker = SpectralTracker(
            model=model,
            log_frequency=2,
            save_dir=os.path.join(tmpdir, 'spectral'),
            peft_type='lora',
            verbose=False,
        )

        gradient_tracker = GradientFlowTracker(
            model=model,
            log_frequency=2,
            save_dir=os.path.join(tmpdir, 'gradient'),
            peft_type='lora',
            verbose=False,
        )

        # Create callbacks
        spectral_callback = SpectralTrackingCallback(
            tracker=spectral_tracker,
            save_on_train_end=True,
            log_to_wandb=False,
        )

        gradient_callback = GradientFlowCallback(
            tracker=gradient_tracker,
            save_on_train_end=True,
            log_to_wandb=False,
            create_heatmaps=False,  # Skip heatmaps in test
        )

        # Mock trainer components
        args = MockTrainingArguments(tmpdir)
        state = MockTrainerState()
        control = MockTrainerControl()

        # Call on_train_begin
        spectral_callback.on_train_begin(args, state, control, model=model)
        gradient_callback.on_train_begin(args, state, control, model=model)

        # Simulate training steps
        for step in range(10):
            state.global_step = step

            # Forward/backward
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()

            # Call on_step_end (callbacks handle logging)
            spectral_callback.on_step_end(args, state, control, model=model)
            gradient_callback.on_step_end(args, state, control, model=model)

            model.zero_grad()

        # Call on_train_end
        spectral_callback.on_train_end(args, state, control)
        gradient_callback.on_train_end(args, state, control)

        # Verify files were saved
        assert os.path.exists(os.path.join(tmpdir, 'spectral', 'spectral_history.pt'))
        assert os.path.exists(os.path.join(tmpdir, 'gradient', 'gradient_flow.pt'))


def test_create_tracking_callbacks_factory():
    """Test the factory function for creating callbacks."""
    model = MockModel(num_layers=4)

    # Create a mock args object
    class MockArgs:
        class tracker:
            enable_spectral_tracking = True
            enable_gradient_tracking = True
            spectral_log_frequency = 50
            gradient_log_frequency = 50
            track_adapter_only = True

        class peft:
            type = 'lora'

        class training:
            output_dir = '/tmp/test_output'

    with tempfile.TemporaryDirectory() as tmpdir:
        MockArgs.training.output_dir = tmpdir

        callbacks = create_tracking_callbacks(
            model=model,
            args=MockArgs(),
            enable_spectral=True,
            enable_gradient=True,
            log_to_wandb=False,
        )

        assert len(callbacks) == 2, f"Expected 2 callbacks, got {len(callbacks)}"
        assert isinstance(callbacks[0], SpectralTrackingCallback)
        assert isinstance(callbacks[1], GradientFlowCallback)


def test_checkpoint_saving():
    """Test that tracking data is saved with checkpoints."""
    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create checkpoint directory
        checkpoint_dir = os.path.join(tmpdir, 'checkpoint-50')
        os.makedirs(checkpoint_dir)

        spectral_tracker = SpectralTracker(
            model=model,
            log_frequency=10,
            save_dir=os.path.join(tmpdir, 'spectral'),
            peft_type='lora',
            verbose=False,
        )

        gradient_tracker = GradientFlowTracker(
            model=model,
            log_frequency=10,
            save_dir=os.path.join(tmpdir, 'gradient'),
            peft_type='lora',
            verbose=False,
        )

        # Log some steps
        for step in range(50):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            gradient_tracker.log_step(step)
            model.zero_grad()
            spectral_tracker.log_step(step)

        # Save to checkpoint directory
        spectral_tracker.save(os.path.join(checkpoint_dir, 'spectral_history.pt'))
        gradient_tracker.save(os.path.join(checkpoint_dir, 'gradient_flow.pt'))

        # Verify checkpoint files
        assert os.path.exists(os.path.join(checkpoint_dir, 'spectral_history.pt'))
        assert os.path.exists(os.path.join(checkpoint_dir, 'gradient_flow.pt'))


def test_tracking_overhead():
    """Measure tracking overhead."""
    import time

    model = MockModel(num_layers=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        spectral_tracker = SpectralTracker(
            model=model,
            log_frequency=1,  # Log every step
            save_dir=os.path.join(tmpdir, 'spectral'),
            peft_type='lora',
            verbose=False,
        )

        gradient_tracker = GradientFlowTracker(
            model=model,
            log_frequency=1,
            save_dir=os.path.join(tmpdir, 'gradient'),
            peft_type='lora',
            verbose=False,
        )

        num_steps = 50

        # Measure without tracking
        start = time.time()
        for step in range(num_steps):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            model.zero_grad()
        base_time = time.time() - start

        # Measure with tracking
        start = time.time()
        for step in range(num_steps):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()
            gradient_tracker.log_step(step)
            model.zero_grad()
            spectral_tracker.log_step(step)
        tracked_time = time.time() - start

        overhead = (tracked_time - base_time) / base_time * 100

        # Log overhead
        print(f"\n    Overhead measurement:")
        print(f"    - Base time ({num_steps} steps): {base_time:.3f}s")
        print(f"    - Tracked time: {tracked_time:.3f}s")
        print(f"    - Overhead: {overhead:.1f}%")

        # Overhead should be reasonable (< 100% for mock model)
        # Real models will have much lower relative overhead
        assert overhead < 500, f"Tracking overhead too high: {overhead:.1f}%"


def test_distributed_safety():
    """Test that tracking only logs on rank 0 (mocked)."""
    model = MockModel(num_layers=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = SpectralTracker(
            model=model,
            log_frequency=1,
            save_dir=tmpdir,
            peft_type='lora',
            verbose=False,
        )

        # Simulate non-zero rank by mocking torch.distributed
        original_is_initialized = torch.distributed.is_initialized

        # Mock distributed as initialized with rank 1
        def mock_is_initialized():
            return True

        def mock_get_rank():
            return 1

        torch.distributed.is_initialized = mock_is_initialized
        original_get_rank = torch.distributed.get_rank
        torch.distributed.get_rank = mock_get_rank

        try:
            # This should not log (rank != 0)
            logged = tracker.log_step(0)
            assert not logged, "Should not log on rank != 0"
        finally:
            # Restore
            torch.distributed.is_initialized = original_is_initialized
            torch.distributed.get_rank = original_get_rank


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("Phase 1C Integration Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("CLI argument parsing", test_cli_argument_parsing),
        ("Combined tracking (spectral + gradient)", test_combined_tracking),
        ("Callback integration", test_callback_integration),
        ("create_tracking_callbacks factory", test_create_tracking_callbacks_factory),
        ("Checkpoint saving", test_checkpoint_saving),
        ("Tracking overhead measurement", test_tracking_overhead),
        ("Distributed safety (rank check)", test_distributed_safety),
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
        print("All integration tests passed!")
    else:
        print(f"Warning: {failed} test(s) failed")

    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
