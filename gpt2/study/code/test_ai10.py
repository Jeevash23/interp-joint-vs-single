"""Unit tests for AI-10 (CPU, toy tensors and the tokeniser only; no pretrained model is loaded).

    CUDA_VISIBLE_DEVICES= python -B study/code/test_ai10.py

1. prompts: layout, answers and counterfactual inputs of every development and confirmation draw.
2. census: AI-05's constructed example (paper Section 3) reproduces its saved counts
   (AI-05 analysis/results/constructed-example.json).
3. margin rule: the split J = S_A(z_r', s) + u_A holds exactly; u_A equals AI-05's form when P_A P_B = 0 and differs
   by (z_t - z_r) P_B P_A otherwise; for a margin linear in the state, the first-order change is exact.
4. DAS: on a toy network whose answer is (a + b) mod 3, read from a planted direction and a planted plane, joint and
   separate fits recover them and pass every single and joint query.
5. GPT-2 code: equals transformers' GPT2LMHeadModel on a tiny random configuration; the final-position path equals
   the full path; null edits reproduce natural margins at every site.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import torch

import ai10_common as common
import das
import interchange as ic
import margin_rule as mr
import prompts

torch.set_num_threads(2)
PASSED = []


def test(fn):
    fn()
    PASSED.append(fn.__name__)
    print("ok", fn.__name__, flush=True)
    return fn


@test
def prompts_domain():
    tok = prompts.load_tokenizer()
    for name in ("development-draws.json", "confirmation-draws.json"):
        tasks = json.loads((common.PROMPTS / name).read_text())
        prompts.check(tok, tasks)
    x = prompts.inputs(3)
    dom = ic.Domain(x, prompts.answer_class(x[:, 0], x[:, 1]))
    assert dom.n == 18 and len(dom.triples) == 5832
    # Every single and joint target is in the domain and keeps the recipient's template.
    for v in (0, 1):
        t = dom.single_target(v)
        assert (x[t, 2] == x[dom.pairs[:, 0], 2]).all() and (x[t, v] == x[dom.pairs[:, 1], v]).all()
    jt = dom.joint_target()
    assert (x[jt, 0] == x[dom.triples[:, 1], 0]).all() and (x[jt, 1] == x[dom.triples[:, 2], 1]).all()
    # Every single query that changes its variable changes the answer; in the joint panel half of the queries
    # predict the recipient's own answer and half the third name.
    jp = dom.panels()["joint"]
    assert (dom.cls[jt][jp] == dom.cls[dom.triples[jp, 0]]).mean() == 0.5
    for v in (0, 1):
        fs = dom.panels()[("full_single", v)]
        assert (dom.cls[dom.single_target(v)][fs] != dom.cls[dom.pairs[fs, 0]]).all()
    sizes = {k if isinstance(k, str) else f"{k[0]}_{k[1]}": int(v.sum()) for k, v in dom.panels().items()}
    assert sizes == {"full_single_0": 162, "minimal_single_0": 18, "full_single_1": 216, "minimal_single_1": 36,
                     "joint": 1944}, sizes


def constructed_example(v_a, v_b, theta):
    """AI-05 analysis/constructed_example.py, through AI-10's census code."""
    x = np.array(list(itertools.product(range(3), repeat=3)))
    s, c = np.sin(theta), np.cos(theta)
    z = np.column_stack([x[:, 0], x[:, 1], np.zeros(27), np.zeros(27)]).astype(float)

    def readout(w):
        relu = lambda u: np.maximum(u, 0.0)  # noqa: E731
        mag = relu(w[:, 2]) + relu(-w[:, 2]) + relu(w[:, 3]) + relu(-w[:, 3])
        return w[:, 0] + w[:, 1] + np.tan(theta) * (w[:, 2] + w[:, 3]) - relu(mag - 2.5 * s * c) / (s * c)

    dom = ic.Domain(x, x[:, 0] + x[:, 1])
    margin = lambda w, rec, target: 1e-4 - np.abs(readout(w) - target)  # noqa: E731  pass iff |error| < 1e-4
    out = ic.census(z, v_a[:, None], v_b[:, None], dom, margin, eps=0.0)
    return out, dom


@test
def census_reproduces_ai05_constructed_example():
    saved = json.loads((common.AI05 / "analysis/results/constructed-example.json").read_text())
    theta = np.deg2rad(saved["theta_degrees"])
    s, c = np.sin(theta), np.cos(theta)
    for key, vb in (("both_tilted", np.array([0, c, 0, s])), ("only_a_tilted", np.array([0, 1.0, 0, 0]))):
        out, dom = constructed_example(np.array([c, 0, s, 0]), vb, theta)
        jp = dom.panels()["joint"]
        got = {"single_failures": int((out["single"] == 1).sum()), "joint_failures": int((out["joint"] == 1).sum()),
               "joint_panel_queries": int(jp.sum()), "joint_panel_failures": int(((out["joint"] == 1) & jp).sum())}
        assert all(got[k] == saved[key][k] for k in got), (key, got)
        split = ic.classify_one_site(out["single"], out["joint"], dom)
        assert split["irreducible"] == saved[key]["joint_panel_failures"] and split["reducible"] == 0


@test
def margin_rule_split_and_ai05_form():
    rng = np.random.default_rng(0)
    x = prompts.inputs(3)
    dom = ic.Domain(x, prompts.answer_class(x[:, 0], x[:, 1]))
    d = 20
    z = rng.normal(size=(dom.n, d))
    w_lin, b_lin = rng.normal(size=d), 0.3
    lin = lambda w: w @ w_lin + b_lin  # noqa: E731   a margin that is linear in the state
    q, _ = np.linalg.qr(rng.normal(size=(d, 4)))
    for fa, fb, orth in ((q[:, :2], q[:, 2:], True), (q[:, :2], np.linalg.qr(q[:, 1:3] + 0.3 * q[:, 3:4])[0], False)):
        sm = np.stack([lin(ic.single_states(z, f, dom.pairs)) for f in (fa, fb)])
        grad = np.tile(w_lin, (dom.n ** 2, 1))
        jm = lin(ic.joint_states(z, fa, fb, dom.triples))
        out = mr.one_site_scores(z, fa, fb, dom, sm, grad, grad, jm, chunk=97)
        assert out["identity_max_abs"] < 1e-12
        ia, ib = dom.reduced()
        assert np.allclose(out["da_hat"], jm - sm[0][ia]) and np.allclose(out["db_hat"], jm - sm[1][ib])
        if orth:
            assert out["ai05_form_max_abs"] < 1e-12
        else:                                   # u_A = AI-05's form + (z_t - z_r) P_B P_A
            r, s_, t = dom.triples.T
            r1 = ia // dom.n
            sb = z[r] + ic.project(z[t] - z[r], fb)
            ua05 = (sb - z[r1]) - ic.project(sb - z[r1], fa)
            extra = ic.project(ic.project(z[t] - z[r], fb), fa)
            ua = ic.joint_states(z, fa, fb, dom.triples) - (z[r1] + ic.project(z[s_] - z[r1], fa))
            assert np.allclose(ua, ua05 + extra) and das.overlap(fa, fb) > 0.1
    assert mr.auroc(np.array([3.0, 2.0, 1.0, 1.0]), np.array([True, False, True, False])) == 0.625


def toy_network(d=12, m=3, seed=1):
    """Answer (a + b) mod 3: a is read from a planted direction e_A, b from a planted plane (e_B1, e_B2) where it sits
    at angle 2 pi b / 3; the readout rotates the b-vector by 2 pi a / 3. n moves the other coordinates."""
    rng = np.random.default_rng(seed)
    x = prompts.inputs(m)
    basis, _ = np.linalg.qr(rng.normal(size=(d, d)))
    e_a, e_b, rest = basis[:, 0], basis[:, 1:3], basis[:, 3:]
    ang = 2 * np.pi * x[:, 1] / 3
    z = (np.outer(x[:, 0], e_a) + np.column_stack([np.cos(ang), np.sin(ang)]) @ e_b.T
         + (rng.normal(size=(m, d - 3)) @ rest.T)[x[:, 2]])
    ea, eb = torch.tensor(e_a, dtype=torch.float32), torch.tensor(e_b, dtype=torch.float32)
    cls_angle = torch.tensor(2 * np.pi * np.arange(3) / 3, dtype=torch.float32)

    def suffix(w, recipients):
        phi = torch.atan2(w @ eb[:, 1], w @ eb[:, 0]) + 2 * np.pi * (w @ ea) / 3
        return 4 * torch.cos(phi[:, None] - cls_angle[None, :])

    def margin(w, recipients, cls):
        import gpt2
        with torch.no_grad():
            return gpt2.margins_from_logits(suffix(torch.tensor(w, dtype=torch.float32), recipients), cls).double().numpy()

    return x, z, e_a, e_b, suffix, margin


@test
def das_recovers_planted_directions():
    x, z, e_a, e_b, suffix, margin = toy_network()
    dom = ic.Domain(x, prompts.answer_class(x[:, 0], x[:, 1]))
    inside = lambda f, e: np.linalg.norm(f.T @ e, axis=0).min()  # noqa: E731   smallest norm of e's projection on f
    (fa, fb), info = das.fit_joint(suffix, z, dom, 2, seed=3, lr=0.05, steps=300, batch=64, device="cpu")
    assert inside(fa, e_a[:, None]) > 0.99 and inside(fb, e_b) > 0.99, (inside(fa, e_a[:, None]), inside(fb, e_b))
    out = ic.census(z, fa, fb, dom, margin, eps=1e-4)
    assert (out["single"] == 0).all() and (out["joint"] == 0).all()
    (sa, sb), _ = das.fit_separate((suffix, suffix), (z, z), dom, 2, seed=5, lr=0.05, steps=300, batch=64,
                                   device="cpu")
    assert inside(sa, e_a[:, None]) > 0.99 and inside(sb, e_b) > 0.99
    ma, mb = das.mean_frames(z, x, orthogonalise=True)
    assert abs(ma.T @ mb).max() < 1e-10 and ma.shape[1] == 1 and mb.shape[1] == 2
    ra, rb = das.random_frames(12, 2, 2, seed=0)
    assert abs(ra.T @ rb).max() < 1e-12


@test
def gpt2_matches_transformers_on_tiny_random_model():
    from transformers import GPT2Config, GPT2LMHeadModel
    import gpt2
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=60, n_positions=16, n_embd=32, n_layer=3, n_head=4, resid_pdrop=0.0,
                     embd_pdrop=0.0, attn_pdrop=0.0)
    ref = GPT2LMHeadModel(cfg).eval()
    tokens = torch.randint(0, 60, (6, 15))
    with torch.no_grad():
        out = ref(tokens, output_hidden_states=True)
    m = gpt2.GPT2(ref.state_dict(), "cpu", heads=4)
    with torch.no_grad():
        nat = m.natural(tokens)
        full = m.ln(nat["resid"][:, -1], *m.lnf) @ m.wte.T
    assert (full - out.logits).abs().max() < 1e-4
    for i in range(cfg.n_layer - 1):                     # HF's hidden_states[i + 1] is the residual after block i
        assert (nat["resid"][:, i] - out.hidden_states[i + 1]).abs().max() < 1e-5
    ids, cls = (5, 9, 17), np.array([0, 1, 2, 1, 2, 0])
    natural_margin = gpt2.margins_from_logits(m.class_logits(nat["resid"][:, -1, gpt2.END], ids), cls)
    with torch.no_grad():
        for positions in (list(range(2, 15)), [gpt2.END]):
            site = gpt2.ResidualSite(m, nat, 0, positions, ids)
            z = torch.tensor(site.natural_states(), dtype=torch.float32)
            got = gpt2.margins_from_logits(site.suffix(z, np.arange(6)), cls)
            assert (got - natural_margin).abs().max() < 1e-4, positions
        heads = gpt2.HeadSites(m, nat, [(1, 1), (1, 3), (2, 0)], (2, 2), ids)
        za, zb = (torch.tensor(a, dtype=torch.float32) for a in heads.natural_states())
        for logits in (heads.suffix_a(za, np.arange(6)), heads.suffix_b(zb, np.arange(6)),
                       heads.forward(np.arange(6), a_states=za, b_frame=torch.eye(8), b_donor=zb)[0]):
            assert (gpt2.margins_from_logits(logits, cls) - natural_margin).abs().max() < 1e-4


@test
def analysis_on_synthetic_rows():
    import analyse
    rows = [{"task": t, "arm": "a", "alignment": k, "single_fail": sf, "single_tie": 0, "joint_fail_all": jf,
             "p_full_single_A": pa, "p_full_single_B": pb, "p_joint": pj, "max_sigma": ms, "flagged_queries": int(ms > 1)}
            for t in range(6) for k, sf, jf, pa, pb, pj, ms in
            [("das_r1", 3, 5, 0.1, 0.0, 0.05, 0.2), ("das_r2", 0, 2, 0.0, 0.0, 0.01, 2.0),
             ("das_r4", 0, 0, 0.0, 0.0, 0.0, 0.5)]]
    jo = analyse.joint_only(rows)
    assert (jo["pass_all_singles"], jo["joint_only"], jo["share"]) == (12, 6, 0.5)
    mr_out = analyse.margin_rule(rows)
    assert mr_out["auroc"] == 1.0 and mr_out["flagged_joint_only"] == 6 and mr_out["flagged_other"] == 0
    det = analyse.detection(rows, reps=200)["120"]
    want = np.mean([[1 - 0.95 ** 30 - (1 - 0.9 ** 20), 1 - 0.99 ** 30, 0.0]])
    assert abs(det["contrast"] - want) < 1e-12 and det["ci95"][0] <= det["contrast"] <= det["ci95"][1]
    assert "contrast_all_single" not in det                                   # older records lack the rates
    for r in rows:                                                             # Section 17's comparison
        r.update(p_all_single_A=r["p_full_single_A"] / 2, p_all_single_B=0.2)
    det = analyse.detection(rows, reps=200)["120"]
    want_all = np.mean([[1 - 0.95 ** 30 - (1 - 0.95 ** 20 * 0.8 ** 20), 1 - 0.99 ** 30 - (1 - 0.8 ** 20),
                         -(1 - 0.8 ** 20)]])
    assert abs(det["contrast_all_single"] - want_all) < 1e-12


@test
def other_scripts_parse():
    import ast
    for name in ("run_study.py", "pilot_checks.py", "overlap_ai05.py", "freeze.py", "gpu_lock.py", "analyse.py"):
        ast.parse((common.CODE / name).read_text())


if __name__ == "__main__":
    print(f"{len(PASSED)} tests passed", file=sys.stderr)
