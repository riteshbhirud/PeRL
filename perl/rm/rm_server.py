import uvicorn
import asyncio
import aiohttp
import logging
import json
import os
import sys
import subprocess
import time
import signal
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

# ==========================================
# 配置区域 (请修改这里)
# ==========================================
# 你的 Qwen 模型路径 (HuggingFace ID 或 本地绝对路径)
MODEL_PATH = "Qwen/Qwen2.5-Math-7B-Instruct" 

# SGLang 服务端口
SGLANG_PORT = 30000
SGLANG_HOST = "0.0.0.0"
SGLANG_URL = f"http://localhost:{SGLANG_PORT}/v1/chat/completions"

# 本 RM 服务端口
RM_SERVER_PORT = 8000

# ==========================================
# 1. 核心算法逻辑 (Math Verify & Utils)
# ==========================================
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("❌ 错误: 请先安装 math-verify。运行 `pip install math-verify`")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AllInOne-RM")

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    if s.startswith(left) and s.endswith("}"):
        return s[len(left) : -1]
    return s

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]

def extract_boxed_answer(solution: str) -> str:
    solution = last_boxed_only_string(solution)
    if solution is None:
        return None
    return remove_boxed(solution)

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> float:
    # 实例化比较器
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    # 格式化输入，确保带上数学环境
    model_output_fmt = "$" + model_output + "$"
    ground_truth_boxed = "$" + ground_truth + "$"
    
    try:
        ret_score, _ = verify_func([ground_truth_boxed], [model_output_fmt])
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass
    return float(ret_score)

# ==========================================
# 2. 进程管理 (自动启动/关闭 SGLang)
# ==========================================

class SGLangManager:
    def __init__(self):
        self.process = None

    def start(self):
        """启动 SGLang 子进程"""
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", MODEL_PATH,
            "--port", str(SGLANG_PORT),
            "--host", SGLANG_HOST,
            # 添加一些优化参数
            "--trust-remote-code" 
        ]
        
        logger.info(f"🚀 正在启动 SGLang 模型服务 (Model: {MODEL_PATH})...")
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        # 启动子进程，将 stdout/stderr 打印到控制台
        self.process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            preexec_fn=os.setsid # 创建新的进程组，方便通过 group id 杀死
        )

    async def wait_until_ready(self):
        """循环检查 SGLang 是否加载完毕"""
        health_url = f"http://localhost:{SGLANG_PORT}/health"
        logger.info("⏳ 等待模型加载 (这可能需要几分钟，请耐心等待)...")
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(health_url) as resp:
                        if resp.status == 200:
                            logger.info("✅ SGLang 模型服务已就绪！")
                            return
                except Exception:
                    pass
                
                # 检查进程是否意外挂了
                if self.process.poll() is not None:
                    logger.error("❌ SGLang 进程意外退出！请检查显存或模型路径。")
                    sys.exit(1)
                
                await asyncio.sleep(5) # 每5秒检查一次

    def stop(self):
        """安全停止 SGLang"""
        if self.process:
            logger.info("🛑 正在停止 SGLang 服务...")
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except Exception as e:
                logger.warning(f"停止进程时遇到问题 (可能已关闭): {e}")
            logger.info("👋 SGLang 服务已关闭")

# 全局管理器实例
sglang_manager = SGLangManager()

# ==========================================
# 3. FastAPI 应用与生命周期
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动阶段 ---
    sglang_manager.start()
    # 阻塞等待，直到 SGLang 加载完成，RM Server 才会开始接收请求
    await sglang_manager.wait_until_ready()
    
    yield # 服务运行中...
    
    # --- 关闭阶段 ---
    sglang_manager.stop()

app = FastAPI(title="All-in-One RM Server", lifespan=lifespan)

class RewardRequest(BaseModel):
    prompt: str
    response: str
    label: str

async def call_qwen_extractor(text: str) -> str:
    """调用后台运行的 SGLang"""
    extraction_prompt = (
        "You are a math answer extractor. Extract the final answer. "
        "Output ONLY the answer inside \\boxed{} if possible, or just the answer/number. "
        "Do not output explanation.\n\n"
        f"Text:\n{text}"
    )
    
    # 为了简化，直接用 requests 风格的 payload，sglang 兼容 OpenAI 格式
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": extraction_prompt}],
        "temperature": 0.0,
        "max_tokens": 128
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(SGLANG_URL, json=payload) as resp:
                if resp.status == 200:
                    res = await resp.json()
                    return res['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"SGLang API Error: {resp.status}")
                    return ""
    except Exception as e:
        logger.error(f"SGLang Call Failed: {e}")
        return ""

@app.post("/reward")
async def calculate_reward(req: RewardRequest):
    # 1. 规则直接提取
    direct_extract = extract_answer(req.response)
    final_ans = direct_extract
    
    # 2. LLM 辅助提取
    if not final_ans:
        qwen_res = await call_qwen_extractor(req.response)
        if qwen_res:
            if "\\boxed" in qwen_res:
                final_ans = extract_boxed_answer(qwen_res)
            else:
                final_ans = qwen_res

    if not final_ans:
        return {"score": 0.0}

    # 3. Math Verify 评分
    score = compute_score(final_ans, req.label)
    
    # 简单的日志
    logger.info(f"GT: {req.label[:20]}... | Extracted: {final_ans} | Score: {score}")
    
    return {"score": score}

@app.get("/health")
def health():
    return {"status": "running"}

if __name__ == "__main__":
    # 启动主服务
    print(f"🔥 正在启动 All-in-One 服务...")
    print(f"👉 HTTP 服务端口: {RM_SERVER_PORT}")
    print(f"👉 SGLang 后台端口: {SGLANG_PORT}")
    
    uvicorn.run(app, host="0.0.0.0", port=RM_SERVER_PORT)