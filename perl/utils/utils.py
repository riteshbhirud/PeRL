import fire
from types import SimpleNamespace
import sys

from perl.config.config import TrainConfig

# System prompt
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def _parse_value(value_str: str):
    """Parse a string value to its appropriate Python type."""
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
    """Set a nested attribute on the config object."""
    parts = config_path.split('.')
    obj = config
    for part in parts[:-1]:
        if isinstance(obj, dict):
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        else:
            if not hasattr(obj, part):
                raise ValueError(f"Unknown config section: {part}")
            obj = getattr(obj, part)

    attr_name = parts[-1]
    if isinstance(obj, dict):
        obj[attr_name] = value
    else:
        if not hasattr(obj, attr_name):
            raise ValueError(f"Unknown config attribute: {attr_name}")
        setattr(obj, attr_name, value)


def parse_args_to_config():
    """
    Parse command line arguments and create TrainConfig.

    Supports two styles of arguments:

    1. Standard nested config: --config.section.attribute value
       Example: --config.tracker.enable_spectral_tracking true

    2. Convenience flags for tracking (Phase 1C):
       --enable_tracking           Enable both spectral and gradient tracking
       --enable_spectral_tracking  Enable only spectral tracking
       --enable_gradient_tracking  Enable only gradient tracking
       --tracking_frequency N      Set tracking frequency for both trackers
       --tracking_output_dir DIR   Override tracking output directory

    Example usage:
        # With tracking enabled (convenience flags)
        python run.py --enable_tracking --tracking_frequency 50 \\
            --config.model.model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

        # With tracking enabled (standard config)
        python run.py --config.tracker.enable_spectral_tracking true \\
            --config.tracker.enable_gradient_tracking true \\
            --config.tracker.spectral_log_frequency 100
    """
    config = TrainConfig()

    # Track convenience flags to apply after parsing
    enable_tracking = False
    enable_spectral = None
    enable_gradient = None
    tracking_frequency = None
    tracking_output_dir = None

    # Parse arguments
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]

        # Convenience flags for tracking (Phase 1C)
        if arg == '--enable_tracking':
            enable_tracking = True
            i += 1
        elif arg == '--enable_spectral_tracking':
            enable_spectral = True
            i += 1
        elif arg == '--enable_gradient_tracking':
            enable_gradient = True
            i += 1
        elif arg == '--tracking_frequency':
            if i + 1 < len(args):
                tracking_frequency = int(args[i + 1])
                i += 2
            else:
                raise ValueError("Missing value for --tracking_frequency")
        elif arg == '--tracking_output_dir':
            if i + 1 < len(args):
                tracking_output_dir = args[i + 1]
                i += 2
            else:
                raise ValueError("Missing value for --tracking_output_dir")

        # Standard nested config arguments
        elif arg.startswith('--config.'):
            config_path = arg[len('--config.'):]
            if i + 1 < len(args):
                value_str = args[i + 1]
                value = _parse_value(value_str)
                _set_nested_attr(config, config_path, value)
                i += 2
            else:
                raise ValueError(f"Missing value for {arg}")
        else:
            i += 1

    # Apply convenience tracking flags
    if enable_tracking:
        config.tracker.enable_spectral_tracking = True
        config.tracker.enable_gradient_tracking = True

    if enable_spectral is not None:
        config.tracker.enable_spectral_tracking = enable_spectral

    if enable_gradient is not None:
        config.tracker.enable_gradient_tracking = enable_gradient

    if tracking_frequency is not None:
        config.tracker.spectral_log_frequency = tracking_frequency
        config.tracker.gradient_log_frequency = tracking_frequency

    if tracking_output_dir is not None:
        # Store for later use in train.py (will be used to override default paths)
        config.tracker.tracking_output_dir = tracking_output_dir

    return config


def print_tracking_info(config):
    """Print tracking configuration summary."""
    tracker = config.tracker

    if not (tracker.enable_spectral_tracking or tracker.enable_gradient_tracking):
        return

    print(f"\n{'='*60}")
    print("MECHANISTIC TRACKING ENABLED")
    print(f"{'='*60}")

    if tracker.enable_spectral_tracking:
        print(f"  Spectral tracking: every {tracker.spectral_log_frequency} steps")
    else:
        print(f"  Spectral tracking: disabled")

    if tracker.enable_gradient_tracking:
        print(f"  Gradient tracking: every {tracker.gradient_log_frequency} steps")
    else:
        print(f"  Gradient tracking: disabled")

    print(f"  Track adapter only: {tracker.track_adapter_only}")
    print(f"  Full SVD: {tracker.compute_full_svd}")

    if hasattr(tracker, 'tracking_output_dir') and tracker.tracking_output_dir:
        print(f"  Output directory: {tracker.tracking_output_dir}")

    print(f"{'='*60}\n")
