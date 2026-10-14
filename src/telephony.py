"""On-the-fly telephony degradation: 16k -> 8k -> G.711 mu-law enc/dec -> 16k. Deterministic, CPU."""
import numpy as np
from scipy.signal import resample_poly

def _mulaw_encode(pcm16):
    x = pcm16.astype(np.int32) >> 2                      # 14-bit
    sign = np.where(x < 0, 0x80, 0).astype(np.int32)
    mag = np.minimum(np.abs(x), 8159) + 33
    seg = np.floor(np.log2(mag)).astype(np.int32) - 5
    mant = (mag >> (seg + 1)) & 0x0F
    return (~(sign | (seg << 4) | mant)) & 0xFF

def _mulaw_decode(code):
    c = (~code.astype(np.int32)) & 0xFF
    seg = (c >> 4) & 7
    t = (((c & 0x0F) << 3) + 0x84) << seg
    return np.where(c & 0x80, 0x84 - t, t - 0x84).astype(np.int16)

def degrade(wav16k: np.ndarray) -> np.ndarray:
    """wav16k: 1-D float array in [-1,1] (or int16). Returns float32 same length (+/-1)."""
    x = np.asarray(wav16k)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    x = x.astype(np.float64)
    n = len(x)
    y8 = resample_poly(x, 1, 2)
    pcm = np.clip(np.round(y8 * 32768.0), -32768, 32767).astype(np.int16)
    dec = _mulaw_decode(_mulaw_encode(pcm)).astype(np.float64) / 32768.0
    y16 = resample_poly(dec, 2, 1)
    if len(y16) < n: y16 = np.pad(y16, (0, n - len(y16)))
    return np.clip(y16[:n], -1, 1).astype(np.float32)
