"""Exhaustive post-freeze analysis of Study 1 (the frozen panel on 80 compiled programs).

For every one of the 480 saved candidates this script evaluates every intervention the
frozen panels could have drawn, instead of the 2-40 sampled queries per panel:

* all 729 recipient/donor pairs for each single-variable interchange (and the minimal-pair
  and restricted-support subsets);
* all 19,683 joint interchanges (recipient, A-donor, B-donor);
* the exact query spaces of the eight frozen panels, so that each candidate's per-query
  failure probability under each panel's sampling rule is known;
* the composition identity (a joint edit equals an A-edit of the B-edited state) and the
  distance of every single-edited state from its matching natural state.

Decisions use the margin rule of the post-freeze amendment: a categorical query fails if the
expected class trails the best alternative by more than 1e-4 and is a tie within 1e-4; a
numerical query fails if it misses by more than 1e-4. Output: posthoc/results/census.json.
It reads saved weights and projectors only and writes nothing else.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"   # set before NumPy loads, as in the frozen study
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "02-Code"))
import numpy as np  # noqa: E402
from v2_model import INPUTS, Network  # noqa: E402

TOL = 1e-4
NAMES = ["compiler", "compiler_null", "mean_full", "mean_restricted", "das_restricted", "random"]
INDEX = {tuple(x): i for i, x in enumerate(INPUTS.tolist())}
RESTRICTED = (INPUTS[:, :2] < 2).all(axis=1)
PAIRS = np.array(list(itertools.product(range(27), repeat=2)))
TRIPLES = np.array(list(itertools.product(range(27), repeat=3)))


def natural(a, b, n):
    return np.array([INDEX[t] for t in zip(a.tolist(), b.tolist(), n.tolist())])


def scores(net, site):
    h = site + np.maximum(site @ net.w1 + net.b1, 0) @ net.w2 + net.b2
    return h[..., net.out_idx]


def decide(net, raw_rows, z_edit, expected):
    """0 = pass, 1 = robust failure, 2 = tie (categorical only)."""
    s = scores(net, net.replace_site(raw_rows, z_edit))
    if not net.categorical:
        return (np.abs(s[:, 0] - expected) > TOL).astype(np.int8)
    k = np.array([net.output_values.index(int(v)) for v in expected])
    rows = np.arange(len(k))
    gap = s[rows, k] - np.where(np.eye(s.shape[1], dtype=bool)[k], -np.inf, s).max(axis=1)
    return np.where(gap > TOL, 0, np.where(gap < -TOL, 1, 2)).astype(np.int8)


def counts(d):
    return {"queries": int(d.size), "fail": int((d == 1).sum()), "tie": int((d == 2).sum())}


def analyse(folder: Path, labels: dict) -> list[dict]:
    net = Network.load(folder)
    _, layers = net.native_numpy(INPUTS)
    raw = layers[net.site][:, -1]
    z = net.encode_site(raw)
    table = np.asarray(net.meta["table"])
    oracle = table[INPUTS[:, 0], INPUTS[:, 1]]
    k = len(net.a_idx) + len(net.b_idx)
    nuis = np.zeros((net.d, net.d))
    nuis[k:, k:] = np.eye(net.d - k)
    keep = np.eye(net.d) - net.rotation.T @ nuis @ net.rotation   # removes the unread nuisance coordinates
    cand = np.load(folder / "candidates.npz")
    # Coordinates outside the edited pool that vary with the input must not be read by the final MLP,
    # so that the output is a function of the edited coordinates alone.
    unedited = [i for i in range(raw.shape[1]) if i not in net.pool and np.ptp(raw[:, i]) > 1e-12]
    unread = float(np.abs(net.w1[unedited]).max()) if unedited else 0.0
    out = []
    r, d = PAIRS.T
    tr, t1, t2 = TRIPLES.T
    jt_nat = natural(INPUTS[t1, 0], INPUTS[t2, 1], INPUTS[tr, 2])           # natural state of the joint target
    for name in NAMES:
        pa, pb = cand[name + "_a"], cand[name + "_b"]
        rec = {"program": folder.name, "family": net.meta["family"], "seed": net.meta["seed"], "candidate": name,
               "valid": labels[name]["reference_consistent"], "varying_unedited_coordinates": len(unedited),
               "final_mlp_weight_on_them": unread}
        single, offs = {}, {}
        for v, p in ((0, pa), (1, pb)):
            tgt = INPUTS[r].copy()
            tgt[:, v] = INPUTS[d, v]
            edit = z[r] + (z[d] - z[r]) @ p
            dec = decide(net, raw[r], edit, table[tgt[:, 0], tgt[:, 1]])
            nat = natural(tgt[:, 0], tgt[:, 1], tgt[:, 2])
            off = np.abs((edit - z[nat]) @ keep).max(axis=1)
            changed = INPUTS[r, v] != INPUTS[d, v]
            other = 1 - v
            minimal = changed & (INPUTS[r, other] == INPUTS[d, other]) & (INPUTS[r, 2] == INPUTS[d, 2])
            restricted = RESTRICTED[r] & RESTRICTED[d]
            heldout = changed & (tgt[:, :2] == 2).any(axis=1)
            single[v] = dec
            offs[v] = off.reshape(27, 27)
            key = "A" if v == 0 else "B"
            rec[f"single_{key}"] = counts(dec)
            rec[f"minimal_{key}"] = counts(dec[minimal])
            rec[f"panel_ordinary_{key}"] = counts(dec[changed & restricted])
            rec[f"panel_full_single_{key}"] = counts(dec[changed])
            rec[f"panel_heldout_single_{key}"] = counts(dec[heldout])
            rec[f"max_off_natural_{key}"] = float(off.max())
        # Joint interchanges: all triples, and the frozen joint stream (both variables change).
        jz = z[tr] + (z[t1] - z[tr]) @ pa + (z[t2] - z[tr]) @ pb
        jdec = decide(net, raw[tr], jz, table[INPUTS[t1, 0], INPUTS[t2, 1]])
        rec["joint"] = counts(jdec)
        stream = (INPUTS[t1, 0] != INPUTS[tr, 0]) & (INPUTS[t2, 1] != INPUTS[tr, 1])
        rec["panel_joint"] = counts(jdec[stream])
        sb = z[tr] + (z[t2] - z[tr]) @ pb
        rec["identity_max_abs"] = float(np.abs(jz - (sb + (z[t1] - sb) @ pa)).max())
        # A joint failure is reducible if a natural single query it reduces to also fails:
        # the A-edit of the natural state (a_r, b_d2, n_r) with donor d1, or the B-edit of (a_d1, b_r, n_r) with donor d2.
        ra = natural(INPUTS[tr, 0], INPUTS[t2, 1], INPUTS[tr, 2])
        rb = natural(INPUTS[t1, 0], INPUTS[tr, 1], INPUTS[tr, 2])
        fail_a = single[0].reshape(27, 27)[ra, t1] == 1
        fail_b = single[1].reshape(27, 27)[rb, t2] == 1
        # Same definition as Study 2: a failure in the joint panel whose two reduced single queries both pass robustly.
        pass_a = single[0].reshape(27, 27)[ra, t1] == 0
        pass_b = single[1].reshape(27, 27)[rb, t2] == 0
        irreducible = (jdec == 1) & stream & pass_a & pass_b
        rec["joint_irreducible_fail"] = int(irreducible.sum())
        rec["joint_panel_fail"] = int(((jdec == 1) & stream).sum())
        # The proposition: an irreducible joint failure needs both single edits to leave the natural states.
        natural_a = offs[0][tr, t1] <= 1e-9
        natural_b = offs[1][tr, t2] <= 1e-9
        rec["irreducible_with_a_natural_edit"] = int((irreducible & (natural_a | natural_b)).sum())
        # Raw latent distance (frozen joint stream) and transport (stream x 27 anchors).
        dist = np.linalg.norm(jz[stream] - z[jt_nat[stream]], axis=1)
        rec["panel_raw_latent"] = {"queries": int(stream.sum()), "fail": int((dist > TOL).sum()), "tie": 0}
        delta = jz[stream] - z[jt_nat[stream]]
        tdec = []
        for anchor in range(27):
            tdec.append(decide(net, np.repeat(raw[[anchor]], len(delta), axis=0), z[anchor] + delta,
                               np.full(len(delta), oracle[anchor])))
        rec["panel_transport"] = counts(np.concatenate(tdec))
        # Complement resampling: replace I - P_A - P_B from the donor; predict the recipient's own output.
        fz = z[r] + (z[d] - z[r]) @ (np.eye(net.d) - pa - pb)
        rec["panel_faithfulness"] = counts(decide(net, raw[r], fz, oracle[r]))
        rec["single_fail_any"] = bool(rec["single_A"]["fail"] or rec["single_B"]["fail"])
        rec["joint_only"] = bool(rec["joint"]["fail"] and not rec["single_fail_any"])
        out.append(rec)
    return out


def same(a, b):
    """Equal up to floating-point round-off (for --check on another machine)."""
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= 1e-9 + 1e-9 * abs(b)
    return a == b


def main():
    rows = []
    for folder in sorted((STUDY / "03-Data/v2/final").iterdir()):
        labels = json.loads((STUDY / f"04-Results/v2/final/{folder.name}.json").read_text())["labels"]
        rows.extend(analyse(folder, labels))
    if "--check" in sys.argv:
        saved = json.loads((STUDY / "posthoc/results/census.json").read_text())
        ok = same(json.loads(json.dumps(rows)), saved)
        print(("ok " if ok else "FAIL") + f" Study 1 census recomputed for {len(rows)} alignments")
        sys.exit(0 if ok else 1)
    dest = STUDY / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "census.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(rows)} candidates; joint-only failures: {sum(x['joint_only'] for x in rows)}; "
          f"labels equal single census: {sum(x['valid'] == (not x['single_fail_any']) for x in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
