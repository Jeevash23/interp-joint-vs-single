"""CPU admission and self-monitoring shared by the bounded v2 jobs."""
import os
import resource
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = time.perf_counter()
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[name] = '2'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2'
os.environ['MPLCONFIGDIR'] = str(ROOT/'04-Results/v2/matplotlib-cache')
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

def available_ram():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))) * 1024

def guard(admission=False):
    available = available_ram()
    free = shutil.disk_usage(ROOT).free
    if available < (3 if admission else 2)*2**30 or free < 10*2**30:
        raise RuntimeError(f'Resource pressure: RAM={available}, disk={free}; stop own worker')
    return {'available_ram_bytes': available, 'disk_free_bytes': free}

def profile():
    return {**guard(), 'elapsed_seconds':time.perf_counter()-START,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'cpu_affinity':sorted(os.sched_getaffinity(0)), 'numerical_thread_limit':2,
            'gpu_used':False, 'scientific_workers':1}
