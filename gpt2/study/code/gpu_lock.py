"""Run a worker while holding the portfolio GPU advisory lock (the lock AI-06 and AI-09 use); refuse if the GPU is busy.

Copied from AI-09 (study/code/gpu_lock.py); only the environment variable's name differs.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

LOCK = Path(__file__).resolve().parents[2] / os.environ.get("GPU_LOCK", "gpu.lock")  # release: GPU_LOCK names the lock file

handle = open(LOCK, "rb")
fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
os.set_inheritable(handle.fileno(), True)
query = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
if int(query.stdout.split(",")[0]) < 4096:
    raise SystemExit("GPU memory is in use by another process; not starting")
print("GPU lock acquired; free/used MiB and temperature:", query.stdout.strip(), flush=True)
os.environ["AI10_GPU_LOCK_FD"] = str(handle.fileno())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.execv(sys.argv[1], sys.argv[1:])
