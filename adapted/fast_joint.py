"""Fast exact re-implementation of joint_decode.joint_beam_search (same search, same scores):
 - CTC prefix extension runs in a numba-compiled loop on CPU (float64) instead of T tiny GPU ops per step
 - LM is queried only for the pre-beam candidates + EOS (the original scored all V pieces then used only these)
 - one GPU->CPU copy per step (no per-candidate .item() syncs)
lm_fn(prefix_tokens_without_sos, cand_list) -> list of log-probs (natural log), cand id == sos means EOS."""
import math, numpy as np, torch
from numba import njit

NEG = -np.inf

@njit(cache=True)
def _lae(a, b):
    if a == -np.inf: return b
    if b == -np.inf: return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))

@njit(cache=True)
def _extend(logp, par, last, cands, empty, out_states, out_scores):
    # logp [T,V]; par [H,T,2]; last [H]; cands [H,C]; empty [H] -> out_states [H,C,T,2], out_scores [H,C]
    T = logp.shape[0]; H, C = cands.shape
    for h in range(H):
        for j in range(C):
            c = cands[h, j]
            r0p = logp[0, c] if empty[h] else -np.inf; r1p = -np.inf
            out_states[h, j, 0, 0] = r0p; out_states[h, j, 0, 1] = r1p
            score = r0p
            for t in range(1, T):
                phi = par[h, t - 1, 1] if c == last[h] else _lae(par[h, t - 1, 0], par[h, t - 1, 1])
                x = logp[t, c]
                score = _lae(score, phi + x)
                r0 = _lae(r0p, phi) + x
                r1 = _lae(r0p, r1p) + logp[t, 0]
                out_states[h, j, t, 0] = r0; out_states[h, j, t, 1] = r1
                r0p = r0; r1p = r1
            out_scores[h, j] = score

class _H:
    __slots__ = ("tokens", "state", "attention", "lm", "score", "ended", "row")
    def __init__(s, tokens, state, attention=0.0, lm=0.0, score=0.0, ended=False):
        s.tokens, s.state, s.attention, s.lm, s.score, s.ended = tokens, state, attention, lm, score, ended

@torch.no_grad()
def fast_joint_beam_search(decode_fn, mem, pad, ctc_logp, sos, beam=12, ctc_weight=0.4, length_bonus=0.1,
                           maxlen=None, pre_beam=None, lm_fn=None, lm_weight=0.0):
    logp = ctc_logp.detach().double().cpu().numpy(); T = logp.shape[0]
    maxlen = T if maxlen is None else maxlen
    pre_beam = max(1, int(1.5 * beam)) if pre_beam is None else pre_beam
    use_lm = lm_fn is not None and lm_weight != 0
    init = np.full((T, 2), NEG); init[:, 1] = np.cumsum(logp[:, 0])
    def full(st): return _lae(st[-1, 0], st[-1, 1])
    def total(a, c, l, n):
        v = length_bonus * n
        if ctc_weight != 1: v += (1 - ctc_weight) * a
        if ctc_weight != 0: v += ctc_weight * c
        if use_lm: v += lm_weight * l
        return v
    hyps = [_H([], init)]; best = None
    for _ in range(maxlen):
        run = [h for h in hyps if not h.ended]
        if not run: break
        ys = torch.tensor([[sos] + h.tokens for h in run], dtype=torch.long, device=mem.device)
        lg = decode_fn(ys, mem.expand(len(run), -1, -1), pad.expand(len(run), -1))
        att_t = torch.log_softmax(lg[:, -1].double(), -1)
        cand_t = att_t[:, 1:sos].topk(min(pre_beam, sos - 1), dim=-1).indices + 1
        att = att_t.cpu().numpy(); cands = cand_t.cpu().numpy().astype(np.int64)
        H, C = cands.shape
        par = np.stack([h.state for h in run]); last = np.array([h.tokens[-1] if h.tokens else -1 for h in run], dtype=np.int64)
        empty = np.array([not h.tokens for h in run]); st = np.empty((H, C, T, 2)); sc = np.empty((H, C))
        _extend(logp, par, last, cands, empty, st, sc)
        pool = [h for h in hyps if h.ended]
        for i, h in enumerate(run):
            lmv = lm_fn(h.tokens, list(cands[i]) + [sos]) if use_lm else None
            for j in range(C):
                tok = int(cands[i, j]); a = h.attention + att[i, tok]; l = h.lm + (lmv[j] if use_lm else 0.0)
                tk = h.tokens + [tok]; pool.append(_H(tk, st[i, j], a, l, total(a, sc[i, j], l, len(tk))))
            a = h.attention + att[i, sos]; l = h.lm + (lmv[C] if use_lm else 0.0)
            pool.append(_H(h.tokens, h.state, a, l, total(a, full(h.state), l, len(h.tokens)), True))
        pool.sort(key=lambda h: h.score, reverse=True); hyps = pool[:beam]
        for h in hyps:
            if h.ended and (best is None or h.score > best.score): best = h
    return list((best or max(hyps, key=lambda h: h.score)).tokens)


# ---------------- incremental (KV-cached) attention decoder: same weights, only the newest token per step -------------
import torch.nn.functional as F

class IncDecoder:
    """Wraps reader.emb / reader.dec (nn.TransformerDecoder, norm_first, gelu) / reader.dec_out for step-wise decoding.
    Memory K/V for cross-attention are projected once per utterance; self-attention K/V are cached per hypothesis."""
    def __init__(self, reader, sinusoid, maxlen=1024):
        self.r = reader; self.d = reader.d; self.layers = reader.dec.layers
        self.nh = self.layers[0].self_attn.num_heads; self.dh = self.d // self.nh
        self.pe = sinusoid(maxlen, self.d, next(reader.parameters()).device)
    def _split(self, x):                                         # [B,L,d] -> [B,nh,L,dh]
        return x.view(x.shape[0], x.shape[1], self.nh, self.dh).transpose(1, 2)
    @torch.no_grad()
    def start(self, mem, pad):
        self.mk, self.mv = [], []
        for ly in self.layers:
            W, b = ly.multihead_attn.in_proj_weight, ly.multihead_attn.in_proj_bias; d = self.d
            self.mk.append(self._split(F.linear(mem, W[d:2 * d], b[d:2 * d]))); self.mv.append(self._split(F.linear(mem, W[2 * d:], b[2 * d:])))
        self.mmask = (~pad)[:, None, None, :]                    # [1,1,1,T] True = attend
        self.K = [None] * len(self.layers); self.V = [None] * len(self.layers)
    @torch.no_grad()
    def step(self, tokens, parents, pos):
        """tokens [H] long (input token at position pos), parents [H] long rows of the previous cache (ignored at pos 0).
        Returns log-softmax over the vocab [H, NV] (float32)."""
        H = tokens.shape[0]; d = self.d
        x = (self.r.emb(tokens) * (d ** 0.5) + self.pe[pos])[:, None]          # [H,1,d]
        for i, ly in enumerate(self.layers):
            sa = ly.self_attn; h = ly.norm1(x)
            qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias); q, k, v = qkv.split(d, -1)
            q, k, v = self._split(q), self._split(k), self._split(v)
            if pos == 0: K, V = k, v
            else: K = torch.cat([self.K[i].index_select(0, parents), k], 2); V = torch.cat([self.V[i].index_select(0, parents), v], 2)
            self.K[i], self.V[i] = K, V
            a = F.scaled_dot_product_attention(q, K, V)
            x = x + sa.out_proj(a.transpose(1, 2).reshape(H, 1, d))
            ca = ly.multihead_attn; h = ly.norm2(x); W, b = ca.in_proj_weight, ca.in_proj_bias
            q = self._split(F.linear(h, W[:d], b[:d]))
            a = F.scaled_dot_product_attention(q, self.mk[i].expand(H, -1, -1, -1), self.mv[i].expand(H, -1, -1, -1), attn_mask=self.mmask)
            x = x + ca.out_proj(a.transpose(1, 2).reshape(H, 1, d))
            x = x + ly.linear2(F.gelu(ly.linear1(ly.norm3(x))))
        return torch.log_softmax(self.r.dec_out(x[:, 0]).float(), -1)

@torch.no_grad()
def fast_joint_beam_search2(inc, mem, pad, ctc_logp, sos, beam=12, ctc_weight=0.4, length_bonus=0.1,
                            maxlen=None, pre_beam=None, lm_fn=None, lm_weight=0.0):
    """Same search as fast_joint_beam_search, but the attention decoder runs incrementally (KV cache)."""
    logp = ctc_logp.detach().double().cpu().numpy(); T = logp.shape[0]; dev = mem.device
    maxlen = T if maxlen is None else maxlen
    pre_beam = max(1, int(1.5 * beam)) if pre_beam is None else pre_beam
    use_lm = lm_fn is not None and lm_weight != 0
    init = np.full((T, 2), NEG); init[:, 1] = np.cumsum(logp[:, 0])
    def full(st): return _lae(st[-1, 0], st[-1, 1])
    def total(a, c, l, n):
        v = length_bonus * n
        if ctc_weight != 1: v += (1 - ctc_weight) * a
        if ctc_weight != 0: v += ctc_weight * c
        if use_lm: v += lm_weight * l
        return v
    inc.start(mem, pad)
    hyps = [_H([], init)]; hyps[0].row = 0; best = None
    for pos in range(maxlen):
        run = [h for h in hyps if not h.ended]
        if not run: break
        toks = torch.tensor([h.tokens[-1] if h.tokens else sos for h in run], dtype=torch.long, device=dev)
        par = torch.tensor([h.row for h in run], dtype=torch.long, device=dev)
        att_t = inc.step(toks, par, pos).double()
        cand_t = att_t[:, 1:sos].topk(min(pre_beam, sos - 1), dim=-1).indices + 1
        both = torch.cat([att_t.gather(1, cand_t), att_t[:, sos:sos + 1]], 1)
        att_c = both.cpu().numpy(); cands = cand_t.cpu().numpy().astype(np.int64)
        H, C = cands.shape
        parS = np.stack([h.state for h in run]); last = np.array([h.tokens[-1] if h.tokens else -1 for h in run], dtype=np.int64)
        empty = np.array([not h.tokens for h in run]); st = np.empty((H, C, T, 2)); sc = np.empty((H, C))
        _extend(logp, parS, last, cands, empty, st, sc)
        pool = [h for h in hyps if h.ended]
        for i, h in enumerate(run):
            lmv = lm_fn(h.tokens, list(cands[i]) + [sos]) if use_lm else None
            for j in range(C):
                tok = int(cands[i, j]); a = h.attention + att_c[i, j]; l = h.lm + (lmv[j] if use_lm else 0.0)
                tk = h.tokens + [tok]; n = _H(tk, st[i, j], a, l, total(a, sc[i, j], l, len(tk))); n.row = i; pool.append(n)
            a = h.attention + att_c[i, C]; l = h.lm + (lmv[C] if use_lm else 0.0)
            n = _H(h.tokens, h.state, a, l, total(a, full(h.state), l, len(h.tokens)), True); n.row = i; pool.append(n)
        pool.sort(key=lambda h: h.score, reverse=True); hyps = pool[:beam]
        for h in hyps:
            if h.ended and (best is None or h.score > best.score): best = h
    return list((best or max(hyps, key=lambda h: h.score)).tokens)


@torch.no_grad()
def fast_joint_beam_search3(inc, mem, pad, ctc_logp, sos, beam=12, ctc_weight=0.4, length_bonus=0.1,
                            maxlen=None, pre_beam=None, lm_fn=None, lm_weight=0.0):
    """search2 with vectorised candidate scoring: only the beam survivors become Python objects.
    Ranking is identical to search2 (stable order: ended pool first, then hyp-major / candidate-minor, EOS last)."""
    logp = ctc_logp.detach().double().cpu().numpy(); T = logp.shape[0]; dev = mem.device
    maxlen = T if maxlen is None else maxlen
    pre_beam = max(1, int(1.5 * beam)) if pre_beam is None else pre_beam
    use_lm = lm_fn is not None and lm_weight != 0
    init = np.full((T, 2), NEG); init[:, 1] = np.cumsum(logp[:, 0])
    wa = (1 - ctc_weight) if ctc_weight != 1 else 0.0; wc = ctc_weight if ctc_weight != 0 else 0.0; wl = lm_weight if use_lm else 0.0
    inc.start(mem, pad)
    h0 = _H([], init); h0.row = 0; hyps = [h0]; best = None
    for pos in range(maxlen):
        run = [h for h in hyps if not h.ended]
        if not run: break
        ended = [h for h in hyps if h.ended]
        toks = torch.tensor([h.tokens[-1] if h.tokens else sos for h in run], dtype=torch.long, device=dev)
        par = torch.tensor([h.row for h in run], dtype=torch.long, device=dev)
        att_t = inc.step(toks, par, pos).double()
        cand_t = att_t[:, 1:sos].topk(min(pre_beam, sos - 1), dim=-1).indices + 1
        both = torch.cat([att_t.gather(1, cand_t), att_t[:, sos:sos + 1]], 1)
        att_c = both.cpu().numpy(); cands = cand_t.cpu().numpy().astype(np.int64)
        H, C = cands.shape
        parS = np.stack([h.state for h in run]); last = np.array([h.tokens[-1] if h.tokens else -1 for h in run], dtype=np.int64)
        empty = np.array([not h.tokens for h in run]); st = np.empty((H, C, T, 2)); sc = np.empty((H, C))
        _extend(logp, parS, last, cands, empty, st, sc)
        ctc = np.concatenate([sc, np.array([[_lae(h.state[-1, 0], h.state[-1, 1])] for h in run])], 1)      # [H,C+1]
        A = np.array([h.attention for h in run])[:, None] + att_c
        if use_lm: Lm = np.array([h.lm for h in run])[:, None] + np.array([lm_fn(h.tokens, list(cands[i]) + [sos]) for i, h in enumerate(run)])
        else: Lm = np.zeros_like(A)
        n = np.array([len(h.tokens) for h in run], dtype=np.float64)[:, None] + np.concatenate([np.ones((H, C)), np.zeros((H, 1))], 1)
        tot = length_bonus * n + wa * A + wc * ctc + wl * Lm
        allsc = np.concatenate([np.array([h.score for h in ended]), tot.ravel()])
        order = np.argsort(-allsc, kind="stable")[:beam]
        new = []
        for k in order:
            if k < len(ended): new.append(ended[k]); continue
            i, j = divmod(int(k) - len(ended), C + 1); h = run[i]
            if j == C: x = _H(h.tokens, h.state, A[i, j], Lm[i, j], tot[i, j], True)
            else: x = _H(h.tokens + [int(cands[i, j])], st[i, j].copy(), A[i, j], Lm[i, j], tot[i, j])
            x.row = i; new.append(x)
        hyps = new
        for h in hyps:
            if h.ended and (best is None or h.score > best.score): best = h
    return list((best or max(hyps, key=lambda h: h.score)).tokens)


# ---------------- cross-utterance batching: U utterances searched in lock-step, one decoder call per step ----------------
class BatchIncDecoder(IncDecoder):
    @torch.no_grad()
    def start_batch(self, mems, pads):
        U = len(mems); Tm = max(m.shape[1] for m in mems); d = self.d; dev = mems[0].device
        M = torch.zeros(U, Tm, d, device=dev, dtype=mems[0].dtype); keep = torch.zeros(U, Tm, dtype=torch.bool, device=dev)
        for u, (m, p) in enumerate(zip(mems, pads)): M[u, :m.shape[1]] = m[0]; keep[u, :m.shape[1]] = ~p[0]
        self.mk, self.mv = [], []
        for ly in self.layers:
            W, b = ly.multihead_attn.in_proj_weight, ly.multihead_attn.in_proj_bias
            self.mk.append(self._split(F.linear(M, W[d:2 * d], b[d:2 * d]))); self.mv.append(self._split(F.linear(M, W[2 * d:], b[2 * d:])))
        self.keep = keep[:, None, None, :]
        self.K = [None] * len(self.layers); self.V = [None] * len(self.layers)
    @torch.no_grad()
    def step_batch(self, tokens, parents, utt, pos):
        H = tokens.shape[0]; d = self.d
        x = (self.r.emb(tokens) * (d ** 0.5) + self.pe[pos])[:, None]
        for i, ly in enumerate(self.layers):
            sa = ly.self_attn; h = ly.norm1(x)
            q, k, v = F.linear(h, sa.in_proj_weight, sa.in_proj_bias).split(d, -1)
            q, k, v = self._split(q), self._split(k), self._split(v)
            if pos == 0: K, V = k, v
            else: K = torch.cat([self.K[i].index_select(0, parents), k], 2); V = torch.cat([self.V[i].index_select(0, parents), v], 2)
            self.K[i], self.V[i] = K, V
            x = x + sa.out_proj(F.scaled_dot_product_attention(q, K, V).transpose(1, 2).reshape(H, 1, d))
            ca = ly.multihead_attn; h = ly.norm2(x); W, b = ca.in_proj_weight, ca.in_proj_bias
            q = self._split(F.linear(h, W[:d], b[:d]))
            a = F.scaled_dot_product_attention(q, self.mk[i].index_select(0, utt), self.mv[i].index_select(0, utt), attn_mask=self.keep.index_select(0, utt))
            x = x + ca.out_proj(a.transpose(1, 2).reshape(H, 1, d))
            x = x + ly.linear2(F.gelu(ly.linear1(ly.norm3(x))))
        return torch.log_softmax(self.r.dec_out(x[:, 0]).float(), -1)

@torch.no_grad()
def fast_joint_batch(inc, items, sos, beam=12, ctc_weight=0.4, length_bonus=0.1, pre_beam=None, lm_fns=None, lm_weight=0.0):
    """items = [(mem, pad, ctc_logp)]; same per-utterance search as fast_joint_beam_search3. Returns list of token lists."""
    U = len(items); dev = items[0][0].device
    pre_beam = max(1, int(1.5 * beam)) if pre_beam is None else pre_beam
    use_lm = lm_fns is not None and lm_weight != 0
    wa = (1 - ctc_weight) if ctc_weight != 1 else 0.0; wc = ctc_weight if ctc_weight != 0 else 0.0; wl = lm_weight if use_lm else 0.0
    LP = [it[2].detach().double().cpu().numpy() for it in items]; Ts = [x.shape[0] for x in LP]
    inc.start_batch([it[0] for it in items], [it[1] for it in items])
    hyps, best = [], [None] * U
    for u in range(U):
        init = np.full((Ts[u], 2), NEG); init[:, 1] = np.cumsum(LP[u][:, 0]); h = _H([], init); h.row = 0; hyps.append([h])
    rows_prev = [0] * U                                             # offset of each utterance's rows in the previous global cache
    for pos in range(max(Ts)):
        runs = [[h for h in hyps[u] if not h.ended] if pos < Ts[u] else [] for u in range(U)]
        active = [u for u in range(U) if runs[u]]
        if not active: break
        toks, par, utt = [], [], []
        for u in active:
            for h in runs[u]: toks.append(h.tokens[-1] if h.tokens else sos); par.append(rows_prev[u] + h.row); utt.append(u)
        att_t = inc.step_batch(torch.tensor(toks, device=dev), torch.tensor(par, device=dev), torch.tensor(utt, device=dev), pos).double()
        cand_t = att_t[:, 1:sos].topk(min(pre_beam, sos - 1), dim=-1).indices + 1
        both = torch.cat([att_t.gather(1, cand_t), att_t[:, sos:sos + 1]], 1)
        att_all = both.cpu().numpy(); cand_all = cand_t.cpu().numpy().astype(np.int64)
        off = 0; new_rows = {}
        for u in active:
            run = runs[u]; H = len(run); att_c = att_all[off:off + H]; cands = cand_all[off:off + H]; base = off; off += H
            C = cands.shape[1]; T = Ts[u]; logp = LP[u]; ended = [h for h in hyps[u] if h.ended]
            parS = np.stack([h.state for h in run]); last = np.array([h.tokens[-1] if h.tokens else -1 for h in run], dtype=np.int64)
            empty = np.array([not h.tokens for h in run]); st = np.empty((H, C, T, 2)); sc = np.empty((H, C))
            _extend(logp, parS, last, cands, empty, st, sc)
            ctc = np.concatenate([sc, np.array([[_lae(h.state[-1, 0], h.state[-1, 1])] for h in run])], 1)
            A = np.array([h.attention for h in run])[:, None] + att_c
            if use_lm: Lm = np.array([h.lm for h in run])[:, None] + np.array([lm_fns[u](h.tokens, list(cands[i]) + [sos]) for i, h in enumerate(run)])
            else: Lm = np.zeros_like(A)
            n = np.array([len(h.tokens) for h in run], dtype=np.float64)[:, None] + np.concatenate([np.ones((H, C)), np.zeros((H, 1))], 1)
            tot = length_bonus * n + wa * A + wc * ctc + wl * Lm
            allsc = np.concatenate([np.array([h.score for h in ended]), tot.ravel()])
            new = []
            for k in np.argsort(-allsc, kind="stable")[:beam]:
                if k < len(ended): new.append(ended[k]); continue
                i, j = divmod(int(k) - len(ended), C + 1); h = run[i]
                x = _H(h.tokens, h.state, A[i, j], Lm[i, j], tot[i, j], True) if j == C else _H(h.tokens + [int(cands[i, j])], st[i, j].copy(), A[i, j], Lm[i, j], tot[i, j])
                x.row = i; new.append(x)
            hyps[u] = new; new_rows[u] = base
            for h in new:
                if h.ended and (best[u] is None or h.score > best[u].score): best[u] = h
        rows_prev = [new_rows.get(u, 0) for u in range(U)]
    return [list((best[u] or max(hyps[u], key=lambda h: h.score)).tokens) for u in range(U)]
