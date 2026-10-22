"""AI-05's first-order margin rule (paper Proposition 2 and Section 5; code: AI-05 studies/2-trained-networks/posthoc/
margin_bound.py and studies/3-multi-site/posthoc/signed_rule.py), unchanged except that u is written in a form that
also covers subspaces that are not orthogonal.

One site. A joint query (r, s, t) reduces to the A-query (r', s) and the B-query (r'', t). With
u_A = J - S_A(z_r', s) and u_B = J - S_B(z_r'', t),

    sigma = min(-grad mu_A . u_A / mu_A, -grad mu_B . u_B / mu_B),   rho = min(|grad mu_A . u_A| / mu_A, ...),

each gradient taken at the state of its single query. When P_A P_B = 0, u_A equals AI-05's
(S_B(z_r, t) - z_r')(I - P_A); in general u_A = that + (z_t - z_r) P_B P_A. The rule flags a joint query if sigma > 1,
and an alignment if any joint query whose two reduced single queries pass is flagged.

Two sites (AI-05 Remark 2, Study 3). Only the B-query (r'', t) remains: u = (w - z^B_r'')(I - P_B), where w is the state
that the A-edit of (r, s) produces at B's site; sigma = -grad mu_B . u / mu_B.
"""
from __future__ import annotations

import numpy as np

from interchange import Domain, joint_states, project


def one_site_scores(z, fa, fb, dom: Domain, single_margin, grad_a, grad_b, joint_margin, chunk: int = 1024) -> dict:
    """single_margin [2, N^2]; grad_a, grad_b [N^2, D] at the single A- and B-query states; joint_margin [N^3]."""
    ia, ib = dom.reduced()
    r, s, t = dom.triples.T
    n = len(dom.triples)
    da_hat, db_hat = np.empty(n), np.empty(n)
    identity, ai05_form = 0.0, 0.0
    orthogonal = np.abs(fa.T @ fb).max() < 1e-6
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        j = joint_states(z, fa, fb, dom.triples[sl])
        r1, r2 = ia[sl] // dom.n, ib[sl] // dom.n
        wa = z[r1] + project(z[s[sl]] - z[r1], fa)          # S_A(z_r', s)
        wb = z[r2] + project(z[t[sl]] - z[r2], fb)          # S_B(z_r'', t)
        ua, ub = j - wa, j - wb
        da_hat[sl] = (grad_a[ia[sl]] * ua).sum(1)
        db_hat[sl] = (grad_b[ib[sl]] * ub).sum(1)
        if orthogonal:                                      # AI-05's form of u_A must agree when P_A P_B = 0
            sb = z[r[sl]] + project(z[t[sl]] - z[r[sl]], fb)
            ua05 = (sb - z[r1]) - project(sb - z[r1], fa)
            ai05_form = max(ai05_form, float(np.abs(ua05 - ua).max()))
        identity = max(identity, float(np.abs(wa + ua - j).max()))
    ma, mb = single_margin[0][ia].astype(float), single_margin[1][ib].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.minimum(-da_hat / ma, -db_hat / mb)
        rho = np.minimum(np.abs(da_hat) / ma, np.abs(db_hat) / mb)
    da, db = joint_margin - ma, joint_margin - mb
    return {"sigma": sigma, "rho": rho, "da_hat": da_hat, "db_hat": db_hat,
            "sign_right_a": np.sign(da_hat) == np.sign(da), "sign_right_b": np.sign(db_hat) == np.sign(db),
            "identity_max_abs": identity, "ai05_form_max_abs": ai05_form if orthogonal else None}


def two_site_scores(zb, fb, w, dom: Domain, margin_b, grad_b) -> dict:
    """zb [N, d_B] natural B-site states; w [N^2, d_B] B-site states after each A-edit (index r*N + s);
    margin_b [N^2] and grad_b [N^2, d_B] at the single B-query states."""
    r, s, t = dom.triples.T
    _, ib = dom.reduced()
    r2 = ib // dom.n
    delta = w[r * dom.n + s] - zb[r2]
    u = delta - project(delta, fb)
    d_hat = (grad_b[ib] * u).sum(1)
    mb = margin_b[ib].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"sigma": -d_hat / mb, "rho": np.abs(d_hat) / mb, "d_hat": d_hat}


def alignment_flag(sigma: np.ndarray, reduced_pass: np.ndarray, threshold: float) -> dict:
    """Largest sigma over joint queries whose reduced single queries pass, and how many it flags."""
    q = sigma[reduced_pass]
    q = q[np.isfinite(q)]
    return {"max_sigma": float(q.max()) if len(q) else float("-inf"), "flagged_queries": int((q > threshold).sum())}


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Probability that a positive outranks a negative, ties counted half (Mann-Whitney U / n1 n2)."""
    pos, neg = score[positive], score[~positive]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                                      # average ranks over ties
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))
