"""Label-synchronous joint CTC/attention decoding for a single utterance."""

from dataclasses import dataclass
from typing import Optional

import torch


class _CTCPrefixScore:
    """Incremental CTC forward states, ordered (nonblank, blank).

    A state [T, 2] describes alignments collapsing to exactly one prefix at
    every frame. ``extend`` also returns the probability that the final CTC
    transcript *starts with* each extension, including any later suffix.
    ``full`` instead returns the exact transcript probability, used for EOS.
    """

    def __init__(self, logp):
        self.logp = logp

    def initial(self):
        """Return the empty-prefix state (only all-blank paths)."""
        state = self.logp.new_full((self.logp.shape[0], 2), -torch.inf)
        state[:, 1] = self.logp[:, 0].cumsum(0)
        return state

    def full(self, state):
        return torch.logsumexp(state[-1], dim=-1)

    def extend(self, prefix, state, candidates):
        """Score distinct nonblank labels together; return [C], [T,2,C].

        Repeating the last label requires a preceding blank. The prefix
        probability sums paths at the first frame that emits the new label;
        subsequent frames integrate to one and must not be counted again.
        """
        scores, states = self.extend_batch([prefix], [state], candidates[None])
        return scores[0], states[:, :, 0]

    def extend_batch(self, prefixes, states, candidates):
        """Extend H parents and C candidates in one frame loop.

        Returns scores [H,C] and forward states [T,2,H,C]. Reductions
        still cover only the two CTC states, in the reference order.
        """
        parents = torch.stack(states, dim=2)  # [T,2,H]
        x = self.logp[:, candidates]          # [T,H,C]
        result = x.new_full((x.shape[0], 2, *candidates.shape), -torch.inf)
        phi = torch.logsumexp(parents, dim=1)[:, :, None].expand_as(x)
        last = candidates.new_tensor([p[-1] if p else -1 for p in prefixes])
        phi = torch.where(candidates[None] == last[None, :, None],
                          parents[:, 1, :, None], phi)
        empty = last == -1
        result[0, 0] = torch.where(empty[:, None], x[0], -torch.inf)
        score = result[0, 0].clone()

        # Before the first reachable parent frame, every entry and forward
        # state is -inf. An extension can first enter one frame later.
        # Empty prefixes can emit at frame zero and must continue at one.
        reachable = (parents != -torch.inf).any(dim=1).any(dim=1).nonzero()
        start = (max(1, int(reachable[0, 0]) + 1)
                 if reachable.numel() else x.shape[0])
        if any(not p for p in prefixes):
            start = 1
        for t in range(start, x.shape[0]):
            entry = phi[t - 1] + x[t]
            result[t, 0] = torch.logaddexp(result[t - 1, 0],
                                         phi[t - 1]) + x[t]
            result[t, 1] = torch.logsumexp(result[t - 1], dim=0) + self.logp[t, 0]
            score = torch.logaddexp(score, entry)
        return score, result


@dataclass
class _Hypothesis:
    tokens: list[int]
    state: torch.Tensor
    attention: float = 0.0
    lm: float = 0.0
    score: float = 0.0
    ended: bool = False


@torch.no_grad()
def joint_beam_search(decode_fn, mem, pad, ctc_logp, sos, beam=12,
                      ctc_weight=0.4, length_bonus=0.1, maxlen=None,
                      pre_beam=None, lm_fn=None, lm_weight=0.0, nbest=1, nbest_feats=False):
    """Return transcript IDs, excluding blank and the shared SOS/EOS ID.

    ``mem`` and ``pad`` have batch size one. Decoder inputs start with SOS;
    ``lm_fn`` receives the same prefixes, including SOS, and returns next-token
    log probabilities. Attention and LM totals include EOS when emitted;
    the length bonus counts only transcript labels.

    Each step batches all running prefixes. Up to ``pre_beam`` lexical labels
    (default max(1, int(1.5 * beam))) are selected by attention probability.
    EOS is always considered separately using the current state's full CTC
    probability. Ended hypotheses compete for the same beam slots and are
    never expanded. The best retained ended hypothesis is returned, or the
    best running one if none ended before ``maxlen`` decoding steps (default
    T). No forced EOS or post-hoc CTC rescoring is performed.

    Inputs and decoder/LM outputs are moved to ``ctc_logp.device``. The model
    itself must support that device. CTC scores must already be normalized
    log probabilities; forward arithmetic uses at least float32 precision.
    """
    if ctc_logp.ndim != 2 or ctc_logp.shape[0] < 1 or ctc_logp.shape[1] < 2:
        raise ValueError("ctc_logp must have shape [T >= 1, V_ctc >= 2]")
    if not ctc_logp.is_floating_point():
        raise ValueError("ctc_logp must be floating point")
    if sos != ctc_logp.shape[1]:
        raise ValueError("sos must equal V_ctc")
    if mem.ndim != 3 or mem.shape[0] != 1 or pad.shape != mem.shape[:2]:
        raise ValueError("mem/pad must have shapes [1,T,D] and [1,T]")
    if mem.shape[1] != ctc_logp.shape[0] or pad.dtype != torch.bool:
        raise ValueError("mem must match CTC frames and pad must be boolean")
    if not isinstance(beam, int) or beam < 1 or not 0 <= ctc_weight <= 1:
        raise ValueError("beam must be positive and ctc_weight must be in [0,1]")
    maxlen = ctc_logp.shape[0] if maxlen is None else maxlen
    pre_beam = max(1, int(1.5 * beam)) if pre_beam is None else pre_beam
    if not isinstance(maxlen, int) or maxlen < 0:
        raise ValueError("maxlen must be a nonnegative integer")
    if not isinstance(pre_beam, int) or pre_beam < 1:
        raise ValueError("pre_beam must be a positive integer")
    device = ctc_logp.device
    dtype = torch.float64 if ctc_logp.dtype == torch.float64 else torch.float32
    scorer = _CTCPrefixScore(ctc_logp.to(dtype=dtype))
    mem, pad = mem.to(device), pad.to(device)
    hypotheses = [_Hypothesis([], scorer.initial())]
    best_ended: Optional[_Hypothesis] = None
    ended_pool = {}                                   # tokens -> best ended hypothesis (for N-best output)

    def total(attention, ctc, lm, length):
        # Avoid 0 * -inf at pure-attention / pure-CTC endpoints.
        value = length_bonus * length
        if ctc_weight != 1:
            value += (1 - ctc_weight) * attention
        if ctc_weight != 0:
            value += ctc_weight * ctc
        if lm_fn is not None and lm_weight != 0:
            value += lm_weight * lm
        return value

    for _ in range(maxlen):
        running = [h for h in hypotheses if not h.ended]
        if not running:
            break
        prefixes = [[sos] + h.tokens for h in running]
        ys = torch.tensor(prefixes, dtype=torch.long, device=device)
        logits = decode_fn(ys, mem.expand(len(running), -1, -1),
                           pad.expand(len(running), -1))
        if logits.shape != (len(running), ys.shape[1], sos + 1):
            raise ValueError("decode_fn returned an unexpected shape")
        att = torch.log_softmax(logits[:, -1].to(device=device, dtype=dtype), -1)
        lm = None
        if lm_fn is not None and lm_weight != 0:
            lm = lm_fn(prefixes).to(device=device, dtype=dtype)
            if lm.shape != att.shape:
                raise ValueError("lm_fn returned an unexpected shape")
        pool = [h for h in hypotheses if h.ended]
        # Preserve the reference's per-row topk call, including tie handling.
        candidates = torch.stack([
            att[i, 1:sos].topk(min(pre_beam, sos - 1)).indices + 1
            for i in range(len(running))])
        ctc_scores, states = scorer.extend_batch(
            [h.tokens for h in running], [h.state for h in running], candidates)
        for i, h in enumerate(running):
            for j, token in enumerate(candidates[i].tolist()):
                a = h.attention + att[i, token].item()
                l = h.lm + (lm[i, token].item() if lm is not None else 0.0)
                tokens = h.tokens + [token]
                pool.append(_Hypothesis(tokens, states[:, :, i, j], a, l,
                            total(a, ctc_scores[i, j].item(), l, len(tokens))))
            a = h.attention + att[i, sos].item()
            l = h.lm + (lm[i, sos].item() if lm is not None else 0.0)
            pool.append(_Hypothesis(h.tokens, h.state, a, l,
                        total(a, scorer.full(h.state).item(), l, len(h.tokens)), True))
        hypotheses = sorted(pool, key=lambda h: h.score, reverse=True)[:beam]
        for h in hypotheses:
            if h.ended and (best_ended is None or h.score > best_ended.score):
                best_ended = h
            if h.ended:
                key = tuple(h.tokens)
                if key not in ended_pool or h.score > ended_pool[key].score: ended_pool[key] = h
    if nbest > 1:                                      # list of (tokens, total search score), best first
        pool = list(ended_pool.values()) or list(hypotheses)
        pool.sort(key=lambda h: h.score, reverse=True)
        if nbest_feats:                                # separate components for a trained reranker
            return [dict(tokens=list(h.tokens), score=float(h.score), att=float(h.attention), lm=float(h.lm),
                         ctc=float(scorer.full(h.state).item()), length=len(h.tokens)) for h in pool[:nbest]]
        return [(list(h.tokens), float(h.score)) for h in pool[:nbest]]
    return list((best_ended or max(hypotheses, key=lambda h: h.score)).tokens)
