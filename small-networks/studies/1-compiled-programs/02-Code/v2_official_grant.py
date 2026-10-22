"""Run the pinned Grant official script unchanged and capture its exact checks."""
from v2_resources import ROOT,guard,profile
import json,runpy,hashlib
guard(admission=True)
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(2)
path=ROOT/'02-Code/vendor/rep_divergence/divergence/mean_diff_section_4.2_sanity_check.py'
d=runpy.run_path(str(path))
r={k:d[k].tolist() for k in ('sA1','sA2','sB1','sB2','shat_B1','shat_B2','shat_A1','shat_A2','shat_single','delta','hhat_B1_next','hhat_B2_next')}
r.update({'all_official_assertions_pass':True,'source_unmodified':True,
          'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'profile':profile()})
(ROOT/'04-Results/v2/official-grant-reproduction.json').write_text(json.dumps(r,indent=2)+'\n')
