import re
import aiohttp
import asyncio
import random

# from examples.retool.tool_sandbox import PythonSandbox, TOOL_CONFIGS

TOOL_CONFIGS = {
    "tool_concurrency": 256,
    "python_memory_limit": 4 * 1024 * 1024 * 1024,  # 4GB
    "python_complie_timeout": 5,
    "python_run_timeout": 10,
}

SEMAPHORE = asyncio.Semaphore(TOOL_CONFIGS["tool_concurrency"])

def extract_python(resp: str) -> str:
    # Handle ```python code blocks
    if "```python" in resp:
        # Find the last occurrence of ```python...```
        python_pattern = r"```python\s*(.*)?```"
        matches = list(re.finditer(python_pattern, resp, re.DOTALL))
        if matches:
            last_match = matches[-1]
            code = last_match.group(1).strip()
            # if (main_start := re.search(r"if __name__ == ['\"]__main__['\"]:", code)) is not None:
            #     code = code[: main_start.start()].strip() + '\nmain()'
            return code
    return None

async def reward_with_unit_test(args, sample, metric='pass_avg', **kwargs):
    code = extract_python(sample.response)
    if code is None:
        return 0.0

    inputs = sample.metadata["unit_tests"]["inputs"]
    outputs = sample.metadata["unit_tests"]["outputs"]

    # Randomly select 20 test cases if there are more than 20
    if len(inputs) > 20:
        indices = random.sample(range(len(inputs)), 20)
        inputs = [inputs[i] for i in indices]
        outputs = [outputs[i] for i in indices]

    total_timeout = TOOL_CONFIGS["python_run_timeout"] + TOOL_CONFIGS["python_complie_timeout"]
    aiohttp_timeout = aiohttp.ClientTimeout(total=total_timeout)

    sandbox_url = args.code_sandbox_url

    # Create a wrapper function that uses semaphore for each individual code execution
    async def execute_with_semaphore(session, code, input):
        payload = {
            "code": get_guarded_code(code),
            "stdin": input,
            "language": "python",
            "compile_timeout": TOOL_CONFIGS["python_complie_timeout"],
            "run_timeout": TOOL_CONFIGS["python_run_timeout"],
            "memory_limit": TOOL_CONFIGS["python_memory_limit"],
        }

        async with SEMAPHORE:
            # return await sandbox.execute_code(code, input=input)
            try:
                async with session.post(sandbox_url, json=payload) as response:
                    if response.status == 200:
                        resp_json = await response.json()
                        return resp_json["run_result"]["stdout"]
                    else:
                        return f"Error: HTTP {response.status}"
            except asyncio.TimeoutError:
                return f"Error: Code execution timed out after {TOOL_CONFIGS['python_run_timeout']} seconds"
            except aiohttp.ClientError as e:
                return f"Error: Client error {str(e)}"
    
    connector = aiohttp.TCPConnector(limit=128, force_close=False)
    async with aiohttp.ClientSession(connector=connector, timeout=aiohttp_timeout) as session:
        task = []
        for input in inputs:
            task.append(execute_with_semaphore(session, code, input))
        results = await asyncio.gather(*task)
    
    corr = 0
    for result, output in zip(results, outputs):
        corr += result.strip() == output.strip()
    if metric == 'pass_avg':
        return corr / len(inputs) if inputs else 0.0
    elif metric == 'pass_all':
        return 1.0 if corr == len(inputs) else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_guarded_code(code: str) -> str:
    indented_code = "\n".join("    " + line for line in code.split("\n"))
    guarded_code = f"""import sys
import traceback
from io import StringIO
import resource

# Set memory limit (4GB)
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, -1))
except Exception:
    pass

import builtins

# builtins.exit = None
builtins.quit = None

import os

os.environ["OMP_NUM_THREADS"] = "1"

os.kill = None
os.system = None
os.putenv = None
os.remove = None
os.removedirs = None
os.rmdir = None
os.fchdir = None
os.setuid = None
os.fork = None
os.forkpty = None
os.killpg = None
os.rename = None
os.renames = None
os.truncate = None
os.replace = None
os.unlink = None
os.fchmod = None
os.fchown = None
os.chmod = None
os.chown = None
os.chroot = None
os.fchdir = None
os.lchflags = None
os.lchmod = None
os.lchown = None
os.getcwd = None
os.chdir = None

import shutil

shutil.rmtree = None
shutil.move = None
shutil.chown = None

import subprocess

subprocess.Popen = None  # type: ignore

import builtins
builtins.help = None

import sys

sys.modules["ipdb"] = None
sys.modules["joblib"] = None
sys.modules["resource"] = None
sys.modules["psutil"] = None
sys.modules["tkinter"] = None

# Redirect stdout and stderr
old_stdout = sys.stdout
old_stderr = sys.stderr
stdout_capture = StringIO()
stderr_capture = StringIO()
sys.stdout = stdout_capture
sys.stderr = stderr_capture

try:
    # User code
{indented_code}
    
    # Get output
    stdout_output = stdout_capture.getvalue()
    stderr_output = stderr_capture.getvalue()
    
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return result
    result = ""
    if stdout_output:
        result += f"{{stdout_output}}"
    if stderr_output:
        result += f"\\nErrors:\\n{{stderr_output}}"
    
    print(result)
    
except Exception as e:
    # Restore standard output
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Return error information
    error_msg = f"Error: {{str(e)}}\\nTraceback:\\n{{traceback.format_exc()}}"
    print(error_msg)"""
    return guarded_code