"""IOI prompts for AI-10: name and word lists, tasks, the factorial domain and tokenisation checks.

A task is a triple of names (n0, n1, n2) and M = 3 templates. The three ordered pairs (n0, n1), (n1, n2), (n2, n0) can fill
the first clause. The domain is every input x = (a, b, n):

  a, position: 0 if the indirect object (IO) is the first-mentioned name, so the second name repeats (ABBA);
               1 if the IO is the second-mentioned name, so the first name repeats (BABA). Changing a with b and n
               fixed changes only which name repeats; the first clause stays as it is.
  b, token:    which ordered pair fills the first clause: (n_b, n_{b+1}). Changing b with a and n fixed puts other
               names in the clause, so the IO's position stays and its identity changes.
  n, template: which of the task's 3 templates.

The answer is n_{(a + b) mod 3}: it changes when either variable changes, as in AI-05's additive tasks, and a joint
interchange that changes both predicts either the recipient's answer or the third name. All 18 inputs are ordinary
IOI prompts, and every counterfactual input of a single or joint interchange is in the domain. Every prompt has the
same 15-token layout:

  0 <|endoftext|>  1 opener  2 NAME1  3 " and"  4 NAME2  5 motion  6 " to"  7 " the"  8 place  9 ","
  10 NAME_S2  11 transfer  12 " a"  13 object  14 " to"     -> next token: the IO name

Run `python -B prompts.py write` to write the development and confirmation draw sequences to study/prompts/
(tokeniser only; no model is loaded).
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np

from ai10_common import CONFIG, PROMPTS, model_dir, save_json, sha256, utc_now

LENGTH, NAME1, NAME2, S2, END = 15, 2, 4, 10, 14
SITE_POSITIONS = list(range(NAME1, LENGTH))     # every position from the first name onward (Section 4.1)

# Development and held-out lists are disjoint. Names that are not one GPT-2 token with a leading space are dropped
# by valid_names(), deterministically.
DEVELOPMENT_NAMES = ["Mary", "John", "Alice", "Bob", "Sarah", "David", "Emma", "James", "Laura", "Michael", "Anna",
                     "Paul", "Kate", "Mark", "Lisa", "Peter", "Jane", "Tom", "Susan", "Daniel"]
HELDOUT_NAMES = ["Amy", "Andrew", "Anne", "Arthur", "Barbara", "Ben", "Betty", "Brian", "Carl", "Carol", "Charles",
                 "Chris", "Claire", "Dan", "Diana", "Donna", "Edward", "Ellen", "Emily", "Eric", "Frank", "Fred",
                 "Gary", "George", "Grace", "Greg", "Hannah", "Harry", "Helen", "Henry", "Jack", "Jacob", "Jason",
                 "Jeff", "Jennifer", "Jessica", "Jim", "Joe", "Jordan", "Joseph", "Joyce", "Julia", "Karen", "Kevin",
                 "Kim", "Larry", "Leo", "Linda", "Lucy", "Luke", "Maria", "Martin", "Matt", "Megan", "Mike", "Nancy",
                 "Nick", "Oliver", "Pat", "Patricia", "Rachel", "Ray", "Rebecca", "Richard", "Robert", "Roger", "Rose",
                 "Ruth", "Ryan", "Sam", "Sandra", "Scott", "Sean", "Sharon", "Simon", "Sophie", "Steve", "Steven",
                 "Ted", "Thomas", "Tim", "Tony", "Victoria", "Walter", "William", "Adam", "Alex", "Ian", "Jake",
                 "Kelly", "Max"]
WORDS = {
    "development": {"opener": ["When", "After"], "motion": [" went", " walked", " ran"],
                    "place": [" store", " park", " school", " station", " restaurant", " hospital"],
                    "transfer": [" gave", " handed"], "object": [" drink", " book", " ball", " ring", " pen", " cup"]},
    "heldout": {"opener": ["While", "Once"], "motion": [" drove", " came", " moved"],
                "place": [" office", " beach", " library", " garden", " market", " house"],
                "transfer": [" passed", " brought", " sent"],
                "object": [" letter", " gift", " bag", " cake", " key", " hat"]},
}
FIELDS = ("opener", "motion", "place", "transfer", "object")


def inputs(m: int) -> np.ndarray:
    """The domain in a fixed order: index i = (3a + b) m + n; columns a, b, n."""
    return np.array([(a, b, n) for a in range(2) for b in range(3) for n in range(m)])


def answer_class(a, b):
    """Index of the answer in the task's names: (a + b) mod 3."""
    return (np.asarray(a) + np.asarray(b)) % 3


def token_strings(task: dict, a: int, b: int, n: int) -> list[str]:
    names = task["names"]
    first, second = names[b], names[(b + 1) % 3]
    repeated = second if a == 0 else first
    t = dict(zip(FIELDS, task["templates"][n]))
    return ["<|endoftext|>", t["opener"], " " + first, " and", " " + second, t["motion"], " to", " the", t["place"],
            ",", " " + repeated, t["transfer"], " a", t["object"], " to"]


def answer_name(task: dict, a: int, b: int) -> str:
    return task["names"][int(answer_class(a, b))]


# ------------------------------------------------------------------------------------------------ tokeniser
def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(model_dir() / "tokenizer.json"))


def single_token(tok, text: str) -> int | None:
    if text == "<|endoftext|>":
        return CONFIG["model"]["bos_token_id"]
    ids = tok.encode(text, add_special_tokens=False).ids
    return ids[0] if len(ids) == 1 else None


def valid_names(tok, names: list[str]) -> list[str]:
    return [x for x in names if single_token(tok, " " + x) is not None]


def task_token_ids(tok, task: dict) -> np.ndarray:
    """Token ids of every input of a task, shape [N, 15], in the order of inputs(M); checked for layout."""
    rows = []
    for a, b, n in inputs(len(task["templates"])).tolist():
        pieces = token_strings(task, a, b, n)
        ids = [single_token(tok, p) for p in pieces]
        if None in ids:
            raise ValueError(f"not single tokens: {[p for p, i in zip(pieces, ids) if i is None]}")
        text = "".join(pieces[1:])
        if tok.encode(text, add_special_tokens=False).ids != ids[1:]:
            raise ValueError(f"round-trip tokenisation failed for {text!r}")
        rows.append(ids)
    return np.array(rows, dtype=np.int64)


def class_token_ids(tok, task: dict) -> tuple[int, int, int]:
    return tuple(single_token(tok, " " + x) for x in task["names"])


# ------------------------------------------------------------------------------------------------ tasks
def templates(split: str, m: int, rng: np.random.Generator) -> list[list[str]]:
    """m distinct templates drawn uniformly from the split's word lists."""
    words = WORDS[split]
    every = list(itertools.product(*(words[f] for f in FIELDS)))
    pick = rng.choice(len(every), size=m, replace=False)
    return [list(every[i]) for i in sorted(pick.tolist())]


def draw_sequence(tok, split: str, count: int, seed: int) -> list[dict]:
    """Tasks in draw order. Each pass shuffles the split's names and groups neighbours in threes, so names are disjoint
    within a pass; later passes reshuffle. The study uses the first tasks that pass the competence gate (PROTOCOL
    Section 3)."""
    names = valid_names(tok, DEVELOPMENT_NAMES if split == "development" else HELDOUT_NAMES)
    out, k = [], 0
    while len(out) < count:
        order = np.random.default_rng([seed, k]).permutation(len(names))
        for i in range(0, len(order) - 2, 3):
            if len(out) == count:
                break
            rng = np.random.default_rng([seed, k, i])
            out.append({"draw": len(out), "split": split, "names": [names[order[j]] for j in (i, i + 1, i + 2)],
                        "templates": templates(split, CONFIG["templates_per_task"], rng)})
        k += 1
    return out


def check(tok, tasks: list[dict]) -> dict:
    for t in tasks:
        ids = task_token_ids(tok, t)
        assert ids.shape == (6 * len(t["templates"]), LENGTH) and len(set(t["names"])) == 3
        cls_ids = np.array(class_token_ids(tok, t))
        x = inputs(len(t["templates"]))
        answer = cls_ids[answer_class(x[:, 0], x[:, 1])]
        # Two different names fill the clause, one of them repeats at S2, and the answer is the other one.
        assert (ids[:, NAME1] != ids[:, NAME2]).all()
        assert ((ids[:, S2] == ids[:, NAME1]) | (ids[:, S2] == ids[:, NAME2])).all() and (ids[:, S2] != answer).all()
        assert ((ids[:, NAME1] == answer) == (x[:, 0] == 0)).all() and ((ids[:, NAME2] == answer) == (x[:, 0] == 1)).all()
        assert set(np.unique(ids[:, [NAME1, NAME2]])) == set(cls_ids.tolist())
    return {"tasks": len(tasks), "inputs_per_task": 6 * CONFIG["templates_per_task"], "layout_tokens": LENGTH}


def main() -> None:
    if sys.argv[1:] != ["write"]:
        raise SystemExit("usage: python -B prompts.py write")
    tok = load_tokenizer()
    dev = draw_sequence(tok, "development", CONFIG["development_max_draws"], CONFIG["seeds"]["development_draws"])
    conf = draw_sequence(tok, "heldout", CONFIG["confirmation_max_draws"], CONFIG["seeds"]["confirmation_draws"])
    report = {"written_at_utc": utc_now(), "tokenizer_sha256": sha256(model_dir() / "tokenizer.json"),
              "development_names": valid_names(tok, DEVELOPMENT_NAMES),
              "development_names_dropped": sorted(set(DEVELOPMENT_NAMES) - set(valid_names(tok, DEVELOPMENT_NAMES))),
              "heldout_names": valid_names(tok, HELDOUT_NAMES),
              "heldout_names_dropped": sorted(set(HELDOUT_NAMES) - set(valid_names(tok, HELDOUT_NAMES))),
              "development_check": check(tok, dev), "confirmation_check": check(tok, conf),
              "example": "".join(token_strings(dev[0], 0, 0, 0)[1:]) + " -> " + answer_name(dev[0], 0, 0)}
    assert not set(report["development_names"]) & set(report["heldout_names"])
    save_json(PROMPTS / "development-draws.json", dev)
    save_json(PROMPTS / "confirmation-draws.json", conf)
    save_json(PROMPTS / "prompt-check.json", report)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_names")}, indent=1))


if __name__ == "__main__":
    main()
