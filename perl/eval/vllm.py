"""
vLLM Process Management and Async Generation for PeRL Evaluation.

This module provides utilities for:
- Starting/stopping vLLM inference servers
- Health checking and readiness polling
- Async text generation via OpenAI-compatible API
"""

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import aiohttp

logger = logging.getLogger("perl.eval.vllm")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VLLMConfig:
    """Configuration for vLLM server."""
    model: str
    port: int = 8000
    host: str = "localhost"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    max_model_len: int = 131072
    dtype: str = "bfloat16"
    seed: Optional[int] = None
    trust_remote_code: bool = True
    extra_args: List[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


# =============================================================================
# Argument Extraction
# =============================================================================

def extract_vllm_args(args) -> Tuple[List[VLLMConfig], Dict[str, Any]]:
    """
    Extract vLLM configuration from command line arguments.

    Args:
        args: Parsed argparse namespace

    Returns:
        Tuple of (list of VLLMConfig, generation kwargs dict)
    """
    configs = []

    # Create config for each data parallel replica
    dp_size = getattr(args, 'dp_size', 1)
    base_port = getattr(args, 'serve_port', 8000)
    tp_size = getattr(args, 'tp_size', 1)
    model = getattr(args, 'model', None)
    gpu_memory = getattr(args, 'gpu_memory_utilization', 0.95)
    seed = getattr(args, 'seed', None)

    for i in range(dp_size):
        config = VLLMConfig(
            model=model,
            port=base_port + i,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory,
            seed=int(seed) if seed is not None else None,
        )
        configs.append(config)

    # Extract generation kwargs
    gen_kwargs = {
        "temperature": getattr(args, 'temperature', 1.0),
        "top_p": getattr(args, 'top_p', 1.0),
        "max_tokens": getattr(args, 'max_new_tokens', 16384),
    }

    return configs, gen_kwargs


# =============================================================================
# Process Management
# =============================================================================

_vllm_processes: List[subprocess.Popen] = []


def start_vllm_processes(
    configs: List[VLLMConfig],
    log_dir: Optional[str] = None
) -> List[subprocess.Popen]:
    """
    Start vLLM server processes.

    Args:
        configs: List of VLLMConfig for each server
        log_dir: Directory for log files (optional)

    Returns:
        List of Popen process objects
    """
    global _vllm_processes
    processes = []

    for i, config in enumerate(configs):
        logger.info(f"Starting vLLM server {i+1}/{len(configs)} on port {config.port}")

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--port", str(config.port),
            "--host", config.host,
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--dtype", config.dtype,
            "--max-model-len", str(config.max_model_len),
        ]

        if config.trust_remote_code:
            cmd.append("--trust-remote-code")

        if config.seed is not None:
            cmd.extend(["--seed", str(config.seed)])

        # Add extra args
        cmd.extend(config.extra_args)

        # Set CUDA_VISIBLE_DEVICES for this process
        env = os.environ.copy()
        gpu_start = i * config.tensor_parallel_size
        gpu_end = (i + 1) * config.tensor_parallel_size
        gpu_ids = ",".join(str(g) for g in range(gpu_start, gpu_end))
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids

        logger.info(f"  GPU IDs: {gpu_ids}")
        logger.debug(f"  Command: {' '.join(cmd)}")

        # Setup log files
        stdout_file = None
        stderr_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            stdout_file = open(os.path.join(log_dir, f"vllm_{i}_stdout.log"), "w")
            stderr_file = open(os.path.join(log_dir, f"vllm_{i}_stderr.log"), "w")

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_file or subprocess.DEVNULL,
                stderr=stderr_file or subprocess.DEVNULL,
            )
            processes.append(process)
            _vllm_processes.append(process)
            logger.info(f"  Started with PID {process.pid}")
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            # Cleanup already started processes
            stop_vllm_processes(processes)
            raise

    # Register cleanup on exit
    atexit.register(lambda: stop_vllm_processes(_vllm_processes))

    return processes


def stop_vllm_processes(processes: Optional[List[subprocess.Popen]] = None):
    """
    Stop vLLM server processes.

    Args:
        processes: List of processes to stop (uses global list if None)
    """
    global _vllm_processes

    if processes is None:
        processes = _vllm_processes

    for process in processes:
        if process.poll() is None:  # Still running
            logger.info(f"Stopping vLLM server (PID {process.pid})")
            try:
                process.terminate()
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing vLLM server (PID {process.pid})")
                process.kill()
            except Exception as e:
                logger.error(f"Error stopping vLLM server: {e}")

    # Clear global list if we stopped global processes
    if processes is _vllm_processes:
        _vllm_processes = []


async def wait_for_vllm_ready(
    configs: List[VLLMConfig],
    timeout: float = 300,
    poll_interval: float = 5.0
) -> bool:
    """
    Wait for all vLLM servers to become ready.

    Args:
        configs: List of VLLMConfig to check
        timeout: Maximum time to wait in seconds
        poll_interval: Time between health checks

    Returns:
        True if all servers are ready, False if timeout
    """
    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        while time.time() - start_time < timeout:
            all_ready = True

            for config in configs:
                try:
                    async with session.get(
                        config.health_url,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status != 200:
                            all_ready = False
                            break
                except Exception:
                    all_ready = False
                    break

            if all_ready:
                logger.info(f"All {len(configs)} vLLM servers are ready")
                return True

            elapsed = time.time() - start_time
            logger.info(
                f"Waiting for vLLM servers... ({elapsed:.0f}s / {timeout:.0f}s)"
            )
            await asyncio.sleep(poll_interval)

    logger.error(f"Timeout waiting for vLLM servers after {timeout}s")
    return False


# =============================================================================
# Async Generation
# =============================================================================

async def generate_single(
    session: aiohttp.ClientSession,
    config: VLLMConfig,
    prompt: str,
    max_tokens: int = 16384,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Generate a single completion from a vLLM server.

    Args:
        session: aiohttp session
        config: VLLMConfig for the server
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling
        stop: Stop sequences
        **kwargs: Additional generation parameters

    Returns:
        Generated text
    """
    payload = {
        "model": config.model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    if stop:
        payload["stop"] = stop

    # Add any extra kwargs
    payload.update(kwargs)

    try:
        async with session.post(
            config.api_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=600)  # 10 minute timeout
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error(f"vLLM API error: {response.status} - {error_text}")
                return ""

            result = await response.json()
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("text", "")
            return ""

    except asyncio.TimeoutError:
        logger.error(f"Generation timeout for prompt: {prompt[:100]}...")
        return ""
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return ""


async def generate_with_vllm_async(
    configs: List[VLLMConfig],
    prompts: List[str],
    max_tokens: int = 16384,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stop: Optional[List[str]] = None,
    max_concurrent_per_server: int = 16,
    **kwargs
) -> List[str]:
    """
    Generate completions for multiple prompts using vLLM servers.

    Distributes prompts across available servers with rate limiting.

    Args:
        configs: List of VLLMConfig for available servers
        prompts: List of input prompts
        max_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        top_p: Top-p sampling
        stop: Stop sequences
        max_concurrent_per_server: Max concurrent requests per server
        **kwargs: Additional generation parameters

    Returns:
        List of generated texts (same order as prompts)
    """
    if not configs:
        raise ValueError("No vLLM configs provided")

    if not prompts:
        return []

    results = [""] * len(prompts)
    semaphores = [asyncio.Semaphore(max_concurrent_per_server) for _ in configs]

    async def generate_with_semaphore(
        prompt_idx: int,
        prompt: str,
        server_idx: int
    ):
        config = configs[server_idx]
        semaphore = semaphores[server_idx]

        async with semaphore:
            async with aiohttp.ClientSession() as session:
                result = await generate_single(
                    session=session,
                    config=config,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    **kwargs
                )
                results[prompt_idx] = result

    # Create tasks, distributing across servers
    tasks = []
    for i, prompt in enumerate(prompts):
        server_idx = i % len(configs)
        task = asyncio.create_task(generate_with_semaphore(i, prompt, server_idx))
        tasks.append(task)

    # Wait for all tasks
    await asyncio.gather(*tasks, return_exceptions=True)

    return results


# =============================================================================
# Convenience Functions
# =============================================================================

def run_generation_sync(
    model: str,
    prompts: List[str],
    port: int = 8000,
    **gen_kwargs
) -> List[str]:
    """
    Synchronous wrapper for async generation.

    Assumes vLLM server is already running.

    Args:
        model: Model name/path
        prompts: List of prompts
        port: vLLM server port
        **gen_kwargs: Generation parameters

    Returns:
        List of generated texts
    """
    config = VLLMConfig(model=model, port=port)

    async def _run():
        return await generate_with_vllm_async(
            configs=[config],
            prompts=prompts,
            **gen_kwargs
        )

    return asyncio.run(_run())


def check_vllm_available() -> bool:
    """Check if vLLM is available for import."""
    try:
        import vllm
        return True
    except ImportError:
        return False
