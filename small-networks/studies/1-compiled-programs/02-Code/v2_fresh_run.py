"""Create a fresh bounded final-run directory; preserve every original record."""
from v2_resources import ROOT,guard
from pathlib import Path
import tempfile,shutil,subprocess,sys
guard(admission=True)
target=Path(tempfile.mkdtemp(prefix='ai05-v2-reproduction-'))
for directory in ['02-Code','03-Data/v2','04-Results/v2','05-Notes','06-Study-History']:
    (target/directory).mkdir(parents=True,exist_ok=True)
for p in (ROOT/'02-Code').glob('v2_*.py'):shutil.copy2(p,target/'02-Code'/p.name)
for p in (ROOT/'06-Study-History').glob('V2-*'):
    if p.is_file():shutil.copy2(p,target/'06-Study-History'/p.name)
shutil.copy2(ROOT/'05-Notes/V2-QUESTION-COMPARATOR-GAP.md',target/'05-Notes/V2-QUESTION-COMPARATOR-GAP.md')
shutil.copytree(ROOT/'03-Data/v2/development',target/'03-Data/v2/development')
print('Fresh reproduction directory:',target,flush=True)
for name in ['v2_run.py','v2_analyze.py','v2_verify.py']:
    subprocess.run([sys.executable,str(target/'02-Code'/name)],cwd=target,check=True)
print('Reproduction complete:',target,flush=True)
