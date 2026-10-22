"""Post hoc verification of AI-10's GPT-2 Small censuses.

PROTOCOL.md Section 11 planned a NumPy float64 verifier written during development; Section 17 recorded that it was
not, and that it would be labelled post hoc if added. This is that verifier, written on 17 October 2026 after the
freeze, while the confirmation was running. It changes no frozen file and no study output.

An independent implementation of GPT-2 Small in NumPy float64, built from the safetensors weights (no torch, no study
model code), recomputes every task's natural states from its tokens and then the margins of a sample of saved queries
from the saved frames, for arms a, c, a' and b. The sample follows Section 11: every query of every alignment that
fails only jointly, every query whose saved margin is within 0.05 of the tie band, and 2,000 seeded random queries per
phase. A decision disagrees if the recomputed margin falls on the other side of the tie band.

    python -B posthoc/verify.py development|confirmation

Writes posthoc/results/verify-<phase>.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "study"
sys.path.insert(0, str(STUDY / "code"))
import prompts  # noqa: E402   (tokeniser and prompt layout only)

CONFIG = json.loads((STUDY / "config.json").read_text())
END, POS_A, SEED = 14, list(range(2, 15)), 2026101706
HEADS_A, HEAD_B = [(7, 3), (7, 9), (8, 6)], (8, 10)


# ------------------------------------------------------------------------------------------------- model
class NumpyGPT2:
    def __init__(self, path: Path):
        from safetensors.numpy import load_file
        w = {k.removeprefix("transformer."): v.astype(np.float64) for k, v in load_file(str(path)).items()}
        self.wte, self.wpe = w["wte.weight"], w["wpe.weight"]
        self.d, self.h = self.wte.shape[1], 12
        self.dh = self.d // self.h
        self.blocks = [{k: w[f"h.{i}.{k}"] for k in ("ln_1.weight", "ln_1.bias", "attn.c_attn.weight",
                                                     "attn.c_attn.bias", "attn.c_proj.weight", "attn.c_proj.bias",
                                                     "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias",
                                                     "mlp.c_proj.weight", "mlp.c_proj.bias")} for i in range(12)]
        self.lnf = (w["ln_f.weight"], w["ln_f.bias"])

    @staticmethod
    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + 1e-5) * g + b

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def qkv(self, i, x):
        b = self.blocks[i]
        y = self.ln(x, b["ln_1.weight"], b["ln_1.bias"]) @ b["attn.c_attn.weight"] + b["attn.c_attn.bias"]
        q, k, v = np.split(y, 3, axis=-1)
        shape = x.shape[:-1] + (self.h, self.dh)
        return q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def finish(self, i, x, z):
        b = self.blocks[i]
        x = x + z.reshape(x.shape) @ b["attn.c_proj.weight"] + b["attn.c_proj.bias"]
        hdn = self.gelu(self.ln(x, b["ln_2.weight"], b["ln_2.bias"]) @ b["mlp.c_fc.weight"] + b["mlp.c_fc.bias"])
        return x + hdn @ b["mlp.c_proj.weight"] + b["mlp.c_proj.bias"]

    def block(self, i, x):
        q, k, v = self.qkv(i, x)
        t = x.shape[1]
        s = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(self.dh)
        s = np.where(np.triu(np.ones((t, t), bool), 1), -np.inf, s)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        return self.finish(i, x, np.einsum("bhqk,bkhd->bqhd", a, v)), k, v

    def natural(self, tokens):
        x = self.wte[tokens] + self.wpe[: tokens.shape[1]]
        resid, ks, vs, zs = [], [], [], []
        for i in range(12):
            q, k, v = self.qkv(i, x)
            t = x.shape[1]
            s = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(self.dh)
            s = np.where(np.triu(np.ones((t, t), bool), 1), -np.inf, s)
            a = np.exp(s - s.max(-1, keepdims=True))
            a /= a.sum(-1, keepdims=True)
            z = np.einsum("bhqk,bkhd->bqhd", a, v)
            x = self.finish(i, x, z)
            resid.append(x)
            ks.append(k)
            vs.append(v)
            zs.append(z[:, END])
        return np.stack(resid, 1), np.stack(ks, 1), np.stack(vs, 1), np.stack(zs, 1)

    def end_block(self, i, x_end, kc, vc, z_hook=None):
        q, k, v = self.qkv(i, x_end[:, None])
        keys, vals = np.concatenate([kc, k], 1), np.concatenate([vc, v], 1)
        s = np.einsum("bqhd,bkhd->bhqk", q, keys) / np.sqrt(self.dh)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        z = np.einsum("bhqk,bkhd->bqhd", a, vals)[:, 0]
        if z_hook is not None:
            z = z_hook(i, z)
        return self.finish(i, x_end[:, None], z[:, None])[:, 0]

    def margins(self, x_end, ids, cls):
        logits = self.ln(x_end, *self.lnf) @ self.wte[list(ids)].T
        rows = np.arange(len(cls))
        other = np.where(np.eye(3, dtype=bool)[cls], -np.inf, logits).max(1)
        return logits[rows, cls] - other


# ---------------------------------------------------------------------------------------------- domain
def domain(m: int):
    x = np.array([(a, b, n) for a in range(2) for b in range(3) for n in range(m)])
    index = {tuple(v): i for i, v in enumerate(x.tolist())}
    look = lambda a, b, n: np.array([index[k] for k in zip(a, b, n)])  # noqa: E731
    cls = (x[:, 0] + x[:, 1]) % 3
    return x, look, cls


def queries(x, look, kind, idx):
    """Recipient, donors and predicted class of saved query `idx` (AI-05's indexing)."""
    n = len(x)
    if kind in ("single_a", "single_b"):
        v = 0 if kind == "single_a" else 1
        r, d = idx // n, idx % n
        tgt = x[r].copy()
        tgt[:, v] = x[d, v]
        return r, d, None, look(tgt[:, 0], tgt[:, 1], tgt[:, 2])
    r, s, t = idx // (n * n), (idx // n) % n, idx % n
    return r, s, t, look(x[s, 0], x[t, 1], x[r, 2])


def recompute(model, nat, arm, layer, fa, fb, kind, r, s, t, ids, cls):
    resid, ks, vs, zs = nat
    proj = lambda delta, f: (delta @ f) @ f.T  # noqa: E731
    if arm in ("a", "c", "a_end"):
        pos = [END] if arm == "a_end" else POS_A
        base = resid[:, layer]
        z = base[:, pos].reshape(len(base), -1)
        if kind == "single_a":
            e = z[r] + proj(z[s] - z[r], fa)
        elif kind == "single_b":
            e = z[r] + proj(z[s] - z[r], fb)
        else:
            e = z[r] + proj(z[s] - z[r], fa) + proj(z[t] - z[r], fb)
        if arm == "a_end":
            xe = e
            for i in range(layer + 1, 12):
                xe = model.end_block(i, xe, ks[r, i, :END], vs[r, i, :END])
            return model.margins(xe, ids, cls)
        xs = base[r].copy()
        xs[:, pos] = e.reshape(len(r), len(pos), model.d)
        for i in range(layer + 1, 12):
            xs = model.block(i, xs)[0]
        return model.margins(xs[:, END], ids, cls)
    # arm b: heads 7.3, 7.9, 8.6 (site A) and 8.10 (site B) at the final position
    za = np.concatenate([zs[:, l, h] for l, h in HEADS_A], -1)
    zb = zs[:, HEAD_B[0], HEAD_B[1]]
    a_states = None
    if kind in ("single_a", "joint"):
        a_states = za[r] + proj(za[s] - za[r], fa)
    b_states = zb[r] + proj(zb[s] - zb[r], fb) if kind == "single_b" else None
    donor_b = zb[t] if kind == "joint" else None

    def hook(i, z):
        z = z.copy()
        if a_states is not None:
            for j, (l, h) in enumerate(HEADS_A):
                if l == i:
                    z[:, h] = a_states[:, 64 * j:64 * (j + 1)]
        if i == HEAD_B[0]:
            if b_states is not None:
                z[:, HEAD_B[1]] = b_states
            elif donor_b is not None:
                w = z[:, HEAD_B[1]].copy()
                z[:, HEAD_B[1]] = w + proj(donor_b - w, fb)
        return z

    xe = resid[r, 6, END]
    for i in range(7, 12):
        xe = model.end_block(i, xe, ks[r, i, :END], vs[r, i, :END], hook)
    return model.margins(xe, ids, cls)


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    phase = sys.argv[1]
    runs = sorted((STUDY / "results/runs" / phase).glob("*/record.json"))
    model = NumpyGPT2(Path(os.environ.get("MODELS_DIR", ROOT / "models")) / "gpt2-small" / "model.safetensors")  # release
    tok = prompts.load_tokenizer()
    eps = json.loads((STUDY / "results/development-settings.json").read_text())["eps"]
    rng = np.random.default_rng(SEED)
    # Every saved query, so that the random sample is drawn over the whole phase.
    catalogue, joint_only = [], []
    for rec_path in runs:
        rec = json.loads(rec_path.read_text())
        for arm, body in rec["arms"].items():
            arrays = np.load(rec_path.parent / f"arm-{arm}.npz")
            for a in body["alignments"]:
                name = a["alignment"]
                sm, jm = arrays[f"{name}__single_margin"], arrays[f"{name}__joint_margin"]
                if a["single_fail"] == 0 and a["single_tie"] == 0 and a["joint_fail_all"] > 0:
                    joint_only.append((rec_path.parent.name, arm, name))
                for kind, m in (("single_a", sm[0]), ("single_b", sm[1]), ("joint", jm)):
                    catalogue.append((rec_path.parent.name, arm, name, kind, m))
    chosen = {}
    for task, arm, name, kind, m in catalogue:
        near = np.flatnonzero(np.abs(np.abs(m) - eps) < 0.05)
        everything = np.arange(len(m)) if (task, arm, name) in joint_only else np.array([], int)
        chosen[(task, arm, name, kind)] = set(near.tolist()) | set(everything.tolist())
    sizes = np.array([len(c[4]) for c in catalogue])
    for k in rng.choice(len(sizes), size=2000, p=sizes / sizes.sum()):
        task, arm, name, kind, m = catalogue[k]
        chosen[(task, arm, name, kind)].add(int(rng.integers(len(m))))
    out = {"phase": phase, "eps": eps, "queries": 0, "disagreements": [], "max_abs_margin_difference": 0.0,
           "joint_only_alignments": [f"{t}/{a}/{n}" for t, a, n in joint_only], "by_arm": {}}
    for rec_path in runs:
        rec = json.loads(rec_path.read_text())
        spec, layer = rec["task"], rec["settings"]["layer"]
        tokens = prompts.task_token_ids(tok, spec)
        ids = prompts.class_token_ids(tok, spec)
        x, look, _ = domain(len(spec["templates"]))
        nat = model.natural(tokens)
        for arm in rec["arms"]:
            arrays = np.load(rec_path.parent / f"arm-{arm}.npz")
            for a in rec["arms"][arm]["alignments"]:
                name = a["alignment"]
                fa = arrays[f"{name}__fa"].astype(np.float64)
                fb = arrays[f"{name}__fb"].astype(np.float64)
                for kind in ("single_a", "single_b", "joint"):
                    idx = np.array(sorted(chosen[(rec_path.parent.name, arm, name, kind)]), dtype=int)
                    if len(idx) == 0:
                        continue
                    saved_m = (arrays[f"{name}__single_margin"][0 if kind == "single_a" else 1] if kind != "joint"
                               else arrays[f"{name}__joint_margin"])[idx].astype(np.float64)
                    saved_d = (arrays[f"{name}__single"][0 if kind == "single_a" else 1] if kind != "joint"
                               else arrays[f"{name}__joint"])[idx]
                    got = np.concatenate([recompute(model, nat, arm, layer, fa, fb, kind,
                                                    *queries(x, look, kind, idx[i:i + 256])[:3], ids,
                                                    (x[:, 0] + x[:, 1])[queries(x, look, kind, idx[i:i + 256])[3]] % 3)
                                          for i in range(0, len(idx), 256)])
                    dec = np.where(got > eps, 0, np.where(got < -eps, 1, 2))
                    bad = np.flatnonzero((dec != saved_d) & (dec != 2) & (saved_d != 2))
                    stats = out["by_arm"].setdefault(arm, {"queries": 0, "max_abs_margin_difference": 0.0,
                                                           "disagreements": 0})
                    gap = float(np.abs(got - saved_m).max())
                    stats["queries"] += len(idx)
                    stats["max_abs_margin_difference"] = max(stats["max_abs_margin_difference"], gap)
                    stats["disagreements"] += len(bad)
                    out["queries"] += len(idx)
                    out["max_abs_margin_difference"] = max(out["max_abs_margin_difference"], gap)
                    for j in bad[:20]:
                        out["disagreements"].append({"task": rec_path.parent.name, "arm": arm, "alignment": name,
                                                     "kind": kind, "query": int(idx[j]), "saved": float(saved_m[j]),
                                                     "recomputed": float(got[j])})
        print(rec_path.parent.name, json.dumps(out["by_arm"]), flush=True)
    out["passed"] = not out["disagreements"]
    dest = ROOT / "posthoc/results"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"verify-{phase}.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=1))


if __name__ == "__main__":
    main()
