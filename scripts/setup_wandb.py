#!/usr/bin/env python3
"""
WandB Setup Script for PeRL Experiments.

This script helps set up Weights & Biases for monitoring experiments.
Run this once before starting experiments to configure authentication.

Usage:
    python scripts/setup_wandb.py
    python scripts/setup_wandb.py --project peft-rlvr-mechanistic
    python scripts/setup_wandb.py --check  # Just check if logged in
    python scripts/setup_wandb.py --offline  # Set up offline mode
"""

import os
import sys
import argparse
import subprocess


def check_wandb_installed():
    """Check if wandb is installed."""
    try:
        import wandb
        return True, wandb.__version__
    except ImportError:
        return False, None


def check_wandb_logged_in():
    """Check if user is logged in to wandb."""
    try:
        import wandb
        # Try to get the API key
        api = wandb.Api()
        # If we can access the viewer, we're logged in
        viewer = api.viewer
        return True, viewer.username if hasattr(viewer, 'username') else viewer.entity
    except Exception:
        return False, None


def install_wandb():
    """Install wandb via pip."""
    print("Installing wandb...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "wandb"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("Successfully installed wandb")
        return True
    else:
        print(f"Failed to install wandb: {result.stderr}")
        return False


def login_wandb():
    """Interactive login to wandb."""
    try:
        import wandb
        print("\n" + "=" * 60)
        print("WandB Login")
        print("=" * 60)
        print("\nTo log in, you need a WandB API key.")
        print("You can find your API key at: https://wandb.ai/authorize")
        print()

        # Use wandb login which handles interactive input
        wandb.login()

        # Verify login
        logged_in, username = check_wandb_logged_in()
        if logged_in:
            print(f"\nSuccessfully logged in as: {username}")
            return True
        else:
            print("\nLogin verification failed")
            return False

    except Exception as e:
        print(f"Login failed: {e}")
        return False


def setup_offline_mode():
    """Configure wandb for offline mode."""
    print("\n" + "=" * 60)
    print("Setting up WandB Offline Mode")
    print("=" * 60)

    # Set environment variable
    os.environ["WANDB_MODE"] = "offline"

    print("\nOffline mode configured!")
    print("\nTo run in offline mode, either:")
    print("  1. Set environment variable: export WANDB_MODE=offline")
    print("  2. Use config flag: --config.wandb.offline true")
    print("\nTo sync offline runs later:")
    print("  wandb sync output/*/wandb/")

    return True


def verify_project(project: str, entity: str = None):
    """Verify or create a WandB project."""
    try:
        import wandb
        api = wandb.Api()

        # Get entity (username or team)
        if entity is None:
            entity = api.viewer.entity

        print(f"\nChecking project: {entity}/{project}")

        # Try to access the project
        try:
            runs = api.runs(f"{entity}/{project}", per_page=1)
            # If we can access runs, project exists
            print(f"Project '{project}' exists with {len(list(runs))} existing runs")
            return True, entity
        except wandb.errors.CommError:
            # Project doesn't exist yet - that's OK, it will be created on first run
            print(f"Project '{project}' will be created on first run")
            return True, entity

    except Exception as e:
        print(f"Could not verify project: {e}")
        return False, None


def print_usage_instructions(project: str, entity: str):
    """Print instructions for using WandB with PeRL."""
    print("\n" + "=" * 60)
    print("WandB Setup Complete!")
    print("=" * 60)

    print(f"\nYour dashboard URL will be:")
    print(f"  https://wandb.ai/{entity}/{project}")

    print("\n" + "-" * 60)
    print("Running Experiments with WandB:")
    print("-" * 60)

    print("""
1. Basic training with WandB (enabled by default):
   python run.py \\
       --config.model.model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \\
       --config.peft.type lora \\
       --config.wandb.project peft-rlvr-mechanistic

2. Using experiment runner:
   python scripts/run_experiments.py \\
       --config configs/experiments/core_1.5B.yaml

3. Disable WandB (for testing):
   python run.py ... --config.common.debug true

4. Custom WandB settings:
   python run.py \\
       --config.wandb.project my-project \\
       --config.wandb.entity my-team \\
       --config.wandb.tags "custom,tags" \\
       --config.wandb.notes "My experiment description"
""")

    print("-" * 60)
    print("Comparing Runs in Dashboard:")
    print("-" * 60)

    print("""
Filter runs by tags:
  - Click 'Filters' in the Runs table
  - Select 'tags' and choose e.g., 'lora', 'seed42', '1.5B'

Group related runs:
  - Runs are automatically grouped by model size + PEFT type
  - e.g., '1.5B_lora', '7B_dora'

Compare metrics:
  - Select multiple runs in the table
  - Click 'Compare' to see side-by-side charts

View spectral/gradient metrics:
  - Look for 'spectral/*' and 'gradient/*' metrics
  - Create custom charts for tracking over time
""")

    print("-" * 60)
    print("Offline Mode (for unreliable connections):")
    print("-" * 60)

    print("""
1. Enable offline mode:
   export WANDB_MODE=offline
   # OR
   python run.py ... --config.wandb.offline true

2. Run your experiments (data saved locally)

3. Sync when back online:
   wandb sync output/*/wandb/
""")


def main():
    parser = argparse.ArgumentParser(
        description="Set up WandB for PeRL experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--project", "-p",
        type=str,
        default="peft-rlvr-mechanistic",
        help="WandB project name (default: peft-rlvr-mechanistic)"
    )
    parser.add_argument(
        "--entity", "-e",
        type=str,
        default=None,
        help="WandB entity (username or team)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check login status, don't set up"
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set up for offline mode"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PeRL WandB Setup")
    print("=" * 60)

    # Check if wandb is installed
    installed, version = check_wandb_installed()
    if installed:
        print(f"\n✓ WandB installed (version {version})")
    else:
        print("\n✗ WandB not installed")
        response = input("Install wandb? [Y/n] ").strip().lower()
        if response in ['', 'y', 'yes']:
            if not install_wandb():
                sys.exit(1)
        else:
            print("Please install wandb manually: pip install wandb")
            sys.exit(1)

    # Check login status
    logged_in, username = check_wandb_logged_in()
    if logged_in:
        print(f"✓ Logged in as: {username}")
    else:
        print("✗ Not logged in to WandB")

    # Just check mode
    if args.check:
        if logged_in:
            print("\nWandB is ready to use!")
            sys.exit(0)
        else:
            print("\nPlease run setup without --check to log in")
            sys.exit(1)

    # Offline mode setup
    if args.offline:
        setup_offline_mode()
        sys.exit(0)

    # Login if needed
    if not logged_in:
        if not login_wandb():
            print("\nSetup incomplete. Please try again or use --offline mode.")
            sys.exit(1)
        logged_in, username = check_wandb_logged_in()

    # Verify/create project
    entity = args.entity or username
    success, entity = verify_project(args.project, entity)

    if success:
        print_usage_instructions(args.project, entity)
    else:
        print("\nSetup completed with warnings. You may need to create the project manually.")


if __name__ == "__main__":
    main()
