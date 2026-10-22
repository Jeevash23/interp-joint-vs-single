"""A minimal GPT-2 forward pass for AI-10, built from the safetensors weights, with the study's three kinds of site.

Our own code (like AI-05's networks) so that every site is explicit; the pilot checks it against transformers'
GPT2LMHeadModel on the real weights (logits within 1e-4), and the unit tests check it against a tiny random GPT-2.

  ResidualSite  the residual stream after block L at a list of positions: every position from the first name onward
                (arm a, a cut through the network) or the final position only (arm a', the starting design's site).
  HeadSites     the outputs z (before W_O) at the final position of the position heads 7.3, 7.9 and 8.6 (site A,
                192 numbers) and the token head 8.10 (site B, 64 numbers), following MIB (arm b). An A-edit sets the
                three heads' z, computed from the recipient's unedited run; 8.10 is computed from the A-edited run and
                then, in a joint query, its P_B component is replaced with the donor's (AI-05 Remark 2).

Edits at the final position cannot change earlier positions (causal attention), so final-position sites run only the
final position, using the recipient's cached keys and values for the others.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

END = 14


class GPT2:
    def __init__(self, state: dict, device="cpu", heads: int = 12, dtype=torch.float32):
        sd = {k.removeprefix("transformer."): v for k, v in state.items()}
        to = lambda t: t.detach().to(device=device, dtype=dtype)  # noqa: E731
        self.device, self.h = device, heads
        self.wte, self.wpe = to(sd["wte.weight"]), to(sd["wpe.weight"])
        self.d = self.wte.shape[1]
        self.dh = self.d // heads
        self.n_layer = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("h."))
        self.blocks = [{name: to(sd[f"h.{i}.{key}"]) for name, key in
                        [("ln1w", "ln_1.weight"), ("ln1b", "ln_1.bias"), ("qkvw", "attn.c_attn.weight"),
                         ("qkvb", "attn.c_attn.bias"), ("ow", "attn.c_proj.weight"), ("ob", "attn.c_proj.bias"),
                         ("ln2w", "ln_2.weight"), ("ln2b", "ln_2.bias"), ("fcw", "mlp.c_fc.weight"),
                         ("fcb", "mlp.c_fc.bias"), ("pw", "mlp.c_proj.weight"), ("pb", "mlp.c_proj.bias")]}
                       for i in range(self.n_layer)]
        self.lnf = (to(sd["ln_f.weight"]), to(sd["ln_f.bias"]))

    @classmethod
    def from_snapshot(cls, folder, device):
        """Weights and head count from a Hugging Face snapshot folder (any GPT-2 size)."""
        import json
        from safetensors.torch import load_file
        heads = json.loads((folder / "config.json").read_text())["n_head"]
        return cls(load_file(str(folder / "model.safetensors")), device, heads=heads)

    def ln(self, x, w, b):
        return F.layer_norm(x, (self.d,), w, b, 1e-5)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.wte[tokens] + self.wpe[: tokens.shape[1]]

    def _qkv(self, i, x):
        b = self.blocks[i]
        q, k, v = (self.ln(x, b["ln1w"], b["ln1b"]) @ b["qkvw"] + b["qkvb"]).split(self.d, -1)
        shape = x.shape[:-1] + (self.h, self.dh)
        return q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def _mlp(self, i, x):
        b = self.blocks[i]
        return F.gelu(self.ln(x, b["ln2w"], b["ln2b"]) @ b["fcw"] + b["fcb"], approximate="tanh") @ b["pw"] + b["pb"]

    def block(self, i, x, z_hook=None):
        """Full-sequence block. z_hook(i, z) may return a replacement for z [B, T, heads, dh]."""
        q, k, v = self._qkv(i, x)
        t = x.shape[1]
        att = torch.einsum("bqhd,bkhd->bhqk", q, k) / self.dh ** 0.5
        att = att.masked_fill(torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1), float("-inf")).softmax(-1)
        z = torch.einsum("bhqk,bkhd->bqhd", att, v)
        if z_hook is not None:
            z = z_hook(i, z)
        x = x + z.reshape(x.shape) @ self.blocks[i]["ow"] + self.blocks[i]["ob"]
        return x + self._mlp(i, x)

    def run(self, x, start: int, z_hook=None):
        for i in range(start, self.n_layer):
            x = self.block(i, x, z_hook)
        return x

    def natural(self, tokens: torch.Tensor) -> dict:
        """Residual after every block, keys and values of every block, and z at every block, for natural inputs."""
        x = self.embed(tokens)
        resid, keys, values, zs = [], [], [], []
        for i in range(self.n_layer):
            q, k, v = self._qkv(i, x)
            keys.append(k)
            values.append(v)
            store = lambda j, z: (zs.append(z), z)[1]  # noqa: E731
            x = self.block(i, x, store)
            resid.append(x)
        return {"resid": torch.stack(resid, 1), "k": torch.stack(keys, 1), "v": torch.stack(values, 1),
                "z": torch.stack(zs, 1)}                      # [N, layer, T, ...]

    def end_block(self, i, x_end, k_cache, v_cache, z_hook=None):
        """Block i at the final position only; positions before it use the cached keys and values [B, T-1, h, dh]."""
        q, k, v = self._qkv(i, x_end[:, None])
        keys, vals = torch.cat([k_cache, k], 1), torch.cat([v_cache, v], 1)
        att = (torch.einsum("bqhd,bkhd->bhqk", q, keys) / self.dh ** 0.5).softmax(-1)
        z = torch.einsum("bhqk,bkhd->bqhd", att, vals)[:, 0]          # [B, h, dh]
        if z_hook is not None:
            z = z_hook(i, z)
        x = x_end + z.reshape(len(x_end), self.d) @ self.blocks[i]["ow"] + self.blocks[i]["ob"]
        return x + self._mlp(i, x)

    def class_logits(self, x_end, class_ids):
        """Logits of the task's three names at the final position (tied unembedding, no bias)."""
        return self.ln(x_end, *self.lnf) @ self.wte[list(class_ids)].T


def margins_from_logits(logits: torch.Tensor, cls: np.ndarray) -> torch.Tensor:
    """AI-05's margin over the task's classes (its three names): the predicted name's logit minus the largest other."""
    c = torch.as_tensor(np.asarray(cls), device=logits.device)
    rows = torch.arange(len(c), device=logits.device)
    other = logits.masked_fill(torch.nn.functional.one_hot(c, logits.shape[1]).bool(), float("-inf")).max(1).values
    return logits[rows, c] - other


class ResidualSite:
    def __init__(self, model: GPT2, nat: dict, layer: int, positions: list[int], class_ids):
        self.m, self.layer, self.pos, self.ids = model, layer, positions, class_ids
        self.base = nat["resid"][:, layer]                              # [N, T, d] after block `layer`
        self.k, self.v = nat["k"], nat["v"]
        self.end_only = positions == [END]

    def natural_states(self) -> np.ndarray:
        return self.base[:, self.pos].reshape(len(self.base), -1).cpu().double().numpy()

    def suffix(self, states: torch.Tensor, recipients) -> torch.Tensor:
        r = torch.as_tensor(np.asarray(recipients), device=self.base.device)
        if self.end_only:
            x = states
            for i in range(self.layer + 1, self.m.n_layer):
                x = self.m.end_block(i, x, self.k[r, i, :END], self.v[r, i, :END])
            return self.m.class_logits(x, self.ids)
        x = self.base[r].clone()
        x[:, self.pos] = states.reshape(len(r), len(self.pos), self.m.d)
        return self.m.class_logits(self.m.run(x, self.layer + 1)[:, END], self.ids)


class HeadSites:
    """Arm b. Site A: z of heads 7.3, 7.9, 8.6 at the final position; site B: z of 8.10 there (64 numbers each)."""

    def __init__(self, model: GPT2, nat: dict, position_heads, token_head, class_ids):
        self.m, self.ids = model, class_ids
        self.ph, self.th = [tuple(h) for h in position_heads], tuple(token_head)
        self.start = min(layer for layer, _ in self.ph + [self.th])
        self.x0 = nat["resid"][:, self.start - 1, END]                 # residual entering the first edited block
        self.k, self.v = nat["k"], nat["v"]
        z = nat["z"][:, :, END]                                        # [N, layer, heads, dh]
        self.za = torch.cat([z[:, layer, head] for layer, head in self.ph], -1)
        self.zb = z[:, self.th[0], self.th[1]]

    def natural_states(self):
        return self.za.cpu().double().numpy(), self.zb.cpu().double().numpy()

    def forward(self, recipients, a_states=None, b_states=None, b_frame=None, b_donor=None, keep_w=False):
        """a_states [B, 192] set the position heads; b_states [B, 64] set 8.10 outright (single B-query); or b_frame
        with b_donor [B, 64] replaces the P_B component of 8.10's live z (joint query). Returns logits and w."""
        r = torch.as_tensor(np.asarray(recipients), device=self.x0.device)
        seen = {}

        def hook(i, z):
            z = z.clone()
            if a_states is not None:
                dh = self.m.dh
                for j, (layer, head) in enumerate(self.ph):
                    if layer == i:
                        z[:, head] = a_states[:, dh * j:dh * (j + 1)]
            if i == self.th[0]:
                w = z[:, self.th[1]].clone()                 # the live value, before any replacement below
                seen["w"] = w
                if b_states is not None:
                    z[:, self.th[1]] = b_states
                elif b_frame is not None:
                    z[:, self.th[1]] = w + ((b_donor - w) @ b_frame) @ b_frame.T
            return z

        x = self.x0[r]
        for i in range(self.start, self.m.n_layer):
            x = self.m.end_block(i, x, self.k[r, i, :END], self.v[r, i, :END], hook)
        return self.m.class_logits(x, self.ids), (seen.get("w") if keep_w else None)

    def suffix_a(self, states, recipients):
        return self.forward(recipients, a_states=states)[0]

    def suffix_b(self, states, recipients):
        return self.forward(recipients, b_states=states)[0]
