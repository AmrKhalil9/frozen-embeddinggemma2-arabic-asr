"""Is EmbeddingGemma 2's audio tower the Gemma 4 E2B audio tower? Fetches ONLY the audio-tower tensors of the remote
safetensors files by HTTP range request (no full download) and compares them tensor by tensor with the local EmbeddingGemma 2 weights.
Usage: python compare_towers.py <local embeddinggemma-2 model.safetensors> <repo_id> [<repo_id> ...]"""
import json, struct, sys, urllib.request
import numpy as np, torch
from safetensors import safe_open

LOCAL, REPOS = sys.argv[1], sys.argv[2:]
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def rng(url, a, b):
    r = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(r, timeout=120).read()


def remote_header(url):
    n = struct.unpack("<Q", rng(url, 0, 7))[0]
    return json.loads(rng(url, 8, 8 + n - 1)), 8 + n


loc = safe_open(LOCAL, "pt")
lkeys = [k for k in loc.keys() if k.startswith("audio_tower.") or k.startswith("embed_audio.")]
print("local audio tensors:", len(lkeys), "params", sum(loc.get_slice(k).get_shape().numel() if hasattr(loc.get_slice(k).get_shape(), "numel") else int(np.prod(loc.get_slice(k).get_shape())) for k in lkeys) / 1e6, "M")
for repo in REPOS:
    url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
    h, off = remote_header(url); h.pop("__metadata__", None)
    # map remote names to local names by their suffix after the audio-tower prefix
    rmap = {}
    for k in h:
        for pre in ("model.audio_tower.", "audio_tower."):
            if k.startswith(pre): rmap["audio_tower." + k[len(pre):]] = k
        for pre in ("model.embed_audio.", "embed_audio."):
            if k.startswith(pre): rmap["embed_audio." + k[len(pre):]] = k
    common = [k for k in lkeys if k in rmap]
    print(f"\n== {repo}: remote audio tensors {len(rmap)}, name matches {len(common)} / {len(lkeys)} local")
    print("   local-only:", [k for k in lkeys if k not in rmap][:6], " remote-only:", [k for k in rmap if k not in lkeys][:6])
    ident, close, diff, shape = 0, 0, 0, 0; worst = []
    for k in common:
        meta = h[rmap[k]]; a, b = meta["data_offsets"]
        t_r = torch.frombuffer(bytearray(rng(url, off + a, off + b - 1)), dtype=DT[meta["dtype"]]).reshape(meta["shape"]).float()
        t_l = loc.get_tensor(k).float()
        if t_l.shape != t_r.shape: shape += 1; continue
        if torch.equal(t_l, t_r): ident += 1; continue
        rel = ((t_l - t_r).norm() / (t_r.norm() + 1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(t_l.flatten(), t_r.flatten(), dim=0).item()
        if rel < 1e-2: close += 1
        else: diff += 1
        worst.append((rel, cos, k))
    worst.sort(reverse=True)
    print(f"   identical {ident}, nearly identical (rel<1%) {close}, different {diff}, shape mismatch {shape}")
    for rel, cos, k in worst[:8]: print(f"   rel {rel:.4f} cos {cos:.4f}  {k}")
    rels = [w[0] for w in worst]
    if rels: print(f"   median rel diff over non-identical tensors: {np.median(rels):.4f}")
