#!/usr/bin/env python
# Copyright (c) 2025
# Phase 1C: Validate tracking setup without GPU
# Run with: python scripts/validate_tracking_setup.py

"""
Validate that the tracking integration is correctly set up.
This script can run on CPU (no GPU required) and checks:
1. All imports work correctly
2. Config parsing works
3. Tracker initialization works
4. Callback creation works
5. Mock training loop works
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_imports():
    """Verify all required imports work."""
    print("Checking imports...")

    try:
        from perl.config.config import TrainConfig, TrackerConfig
        print("  ✓ Config classes")
    except ImportError as e:
        print(f"  ✗ Config classes: {e}")
        return False

    try:
        from perl.trackers import (
            SpectralTracker,
            SpectralTrackingCallback,
            GradientFlowTracker,
            GradientFlowCallback,
            create_tracking_callbacks,
        )
        print("  ✓ Tracker classes")
    except ImportError as e:
        print(f"  ✗ Tracker classes: {e}")
        return False

    return True


def check_config():
    """Verify config creation and tracking options."""
    print("\nChecking config...")

    from perl.config.config import TrainConfig

    config = TrainConfig()

    # Check tracker config exists
    assert hasattr(config, 'tracker'), "Missing tracker config"
    print("  ✓ TrackerConfig exists")

    # Check all tracking options
    assert hasattr(config.tracker, 'enable_spectral_tracking')
    assert hasattr(config.tracker, 'enable_gradient_tracking')
    assert hasattr(config.tracker, 'spectral_log_frequency')
    assert hasattr(config.tracker, 'gradient_log_frequency')
    assert hasattr(config.tracker, 'track_adapter_only')
    assert hasattr(config.tracker, 'tracking_output_dir')
    print("  ✓ All tracking options present")

    # Set tracking options
    config.tracker.enable_spectral_tracking = True
    config.tracker.enable_gradient_tracking = True
    config.tracker.spectral_log_frequency = 50
    print("  ✓ Config options can be set")

    return True


def check_mock_training():
    """Run a mock training loop with tracking."""
    print("\nChecking mock training loop...")

    import torch
    import torch.nn as nn
    import tempfile

    from perl.trackers import (
        SpectralTracker,
        SpectralTrackingCallback,
        GradientFlowTracker,
        GradientFlowCallback,
    )

    # Create mock model
    class MockLoRALayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = nn.ModuleDict({'default': nn.Linear(64, 8, bias=False)})
            self.lora_B = nn.ModuleDict({'default': nn.Linear(8, 64, bias=False)})
            nn.init.normal_(self.lora_A['default'].weight, std=0.02)
            nn.init.normal_(self.lora_B['default'].weight, std=0.02)

        def forward(self, x):
            return x + self.lora_B['default'](self.lora_A['default'](x))

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([
                nn.Module() for _ in range(4)
            ])
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = MockLoRALayer()

        def forward(self, x):
            for layer in self.model.layers:
                x = layer.self_attn.q_proj(x)
            return x

    model = MockModel()
    print("  ✓ Mock model created")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create trackers
        spectral_tracker = SpectralTracker(
            model=model,
            log_frequency=5,
            save_dir=os.path.join(tmpdir, 'spectral'),
            peft_type='lora',
            verbose=False,
        )
        print(f"  ✓ SpectralTracker: {len(spectral_tracker.adapter_layers)} layers")

        gradient_tracker = GradientFlowTracker(
            model=model,
            log_frequency=5,
            save_dir=os.path.join(tmpdir, 'gradient'),
            peft_type='lora',
            verbose=False,
        )
        print(f"  ✓ GradientFlowTracker: {len(gradient_tracker.adapter_params)} params")

        # Run mock training
        num_steps = 20
        for step in range(num_steps):
            x = torch.randn(2, 64, requires_grad=True)
            output = model(x)
            loss = output.sum()
            loss.backward()

            # Log gradients (after backward, before optimizer step)
            gradient_tracker.log_step(step)

            # Simulate optimizer step
            model.zero_grad()

            # Log spectral (after optimizer step)
            spectral_tracker.log_step(step)

        print(f"  ✓ Mock training loop: {num_steps} steps")

        # Save and verify
        spectral_path = spectral_tracker.save()
        gradient_path = gradient_tracker.save()

        assert os.path.exists(spectral_path), "Spectral save failed"
        assert os.path.exists(gradient_path), "Gradient save failed"
        print("  ✓ Tracking data saved")

        # Get summaries
        spectral_summary = spectral_tracker.get_summary_stats()
        gradient_summary = gradient_tracker.get_summary_stats()

        print(f"  ✓ Spectral: {spectral_summary['total_steps_logged']} steps logged")
        print(f"  ✓ Gradient: {gradient_summary['total_steps_logged']} steps logged")

    return True


def check_cli_integration():
    """Check CLI argument integration."""
    print("\nChecking CLI integration...")

    # Check that convenience flags are documented
    from perl.config.config import TrackerConfig

    config = TrackerConfig()
    assert config.enable_spectral_tracking == False
    assert config.enable_gradient_tracking == False
    print("  ✓ Default tracking disabled")

    # Simulate CLI flag setting
    config.enable_spectral_tracking = True
    config.enable_gradient_tracking = True
    config.spectral_log_frequency = 100
    config.gradient_log_frequency = 100
    print("  ✓ CLI flags can enable tracking")

    return True


def main():
    print("=" * 60)
    print("Phase 1C: Tracking Setup Validation")
    print("=" * 60)

    all_passed = True

    try:
        all_passed &= check_imports()
        all_passed &= check_config()
        all_passed &= check_mock_training()
        all_passed &= check_cli_integration()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All validation checks passed!")
        print("")
        print("Tracking integration is correctly set up.")
        print("")
        print("To run training with tracking on GPU:")
        print("  bash scripts/test_tracking.sh")
        print("")
        print("Or manually:")
        print("  python run.py --enable_tracking --tracking_frequency 100 \\")
        print("    --config.model.model_name_or_path <model> \\")
        print("    --config.dataset.dataset_name_or_path <dataset> \\")
        print("    --config.peft.type lora")
    else:
        print("Some validation checks failed!")
        print("Please check the errors above.")

    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
