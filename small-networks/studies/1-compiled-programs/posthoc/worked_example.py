"""The worked example of the manuscript: lookup program 10000 and its restricted mean alignment.

Recomputes, from the saved weights and projectors, the A-carrier before and after one full-domain single
interchange, the class scores, and how the same alignment fares on the restricted and full single queries.
Output: posthoc/results/worked-example.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / "02-Code"))
import numpy as np  # noqa: E402
from v2_model import INPUTS, Network  # noqa: E402

FOLDER = STUDY / "03-Data/v2/final/lookup-10000"
RECIPIENT, DONOR = (2, 1, 0), (0, 1, 1)


def main():
    net = Network.load(FOLDER)
    _, layers = net.native_numpy(INPUTS)
    raw = layers[net.site][:, -1]
    z = net.encode_site(raw)
    cand = np.load(FOLDER / "candidates.npz")
    pa = cand["mean_restricted_a"]
    idx = {tuple(x): i for i, x in enumerate(INPUTS.tolist())}
    r, d = idx[RECIPIENT], idx[DONOR]
    edited = z[r] + (z[d] - z[r]) @ pa
    site = net.replace_site(raw[[r]], edited[None])
    scores = (site + np.maximum(site @ net.w1 + net.b1, 0) @ net.w2 + net.b2)[0, net.out_idx]
    table = np.asarray(net.meta["table"])
    target = (DONOR[0], RECIPIENT[1])
    census = json.loads((STUDY / "posthoc/results/census.json").read_text())
    rec = next(x for x in census if x["program"] == "lookup-10000" and x["candidate"] == "mean_restricted")
    out = {"table": table.tolist(), "recipient": RECIPIENT, "donor": DONOR,
           "expected_answer": int(table[target]),
           "a_carrier_recipient": raw[r, net.a_idx].round(6).tolist(),
           "a_carrier_donor": raw[d, net.a_idx].round(6).tolist(),
           "a_carrier_after_edit": site[0, net.a_idx].round(6).tolist(),
           "b_carrier_after_edit": site[0, net.b_idx].round(6).tolist(),
           "class_values": net.output_values,
           "class_scores": scores.round(6).tolist(),
           "returned_answer": int(net.output_values[int(np.argmax(scores))]),
           "projector_rank": int(round(np.trace(pa))),
           "restricted_single_failures": rec["panel_ordinary_A"]["fail"] + rec["panel_ordinary_B"]["fail"],
           "restricted_single_queries": rec["panel_ordinary_A"]["queries"] + rec["panel_ordinary_B"]["queries"],
           "full_single_failures": rec["panel_full_single_A"]["fail"] + rec["panel_full_single_B"]["fail"],
           "full_single_queries": rec["panel_full_single_A"]["queries"] + rec["panel_full_single_B"]["queries"]}
    (STUDY / "posthoc/results/worked-example.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
