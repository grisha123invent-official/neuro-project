"""
eeg_utils.py

Утилиты для работы с реальным временем EEG и MNE.
Предполагается sample_rate = 250 Hz (можно менять).
"""

import numpy as np
import mne
from mne.time_frequency import psd_array_welch

SAMPLE_RATE = 250.0

class RingBuffer:
    """Простой кольцевой буфер для N каналов"""
    def __init__(self, n_channels: int, maxlen: int):
        self.n_channels = n_channels
        self.maxlen = int(maxlen)
        self.data = np.zeros((n_channels, self.maxlen), dtype=float)
        self.idx = 0
        self.count = 0

    def append_block(self, block: np.ndarray):
        """
        block: shape (n_channels, n_samples)
        """
        n_ch, n_s = block.shape
        assert n_ch == self.n_channels
        # circularly write
        for i in range(n_s):
            self.data[:, self.idx] = block[:, i]
            self.idx = (self.idx + 1) % self.maxlen
            self.count = min(self.maxlen, self.count + 1)

    def get(self):
        """Return contiguous array (n_channels, n_samples_present) in chronological order."""
        if self.count < self.maxlen:
            return self.data[:, :self.count].copy()
        # wrapped
        out = np.empty((self.n_channels, self.maxlen), dtype=float)
        out[:, :] = np.concatenate((self.data[:, self.idx:], self.data[:, :self.idx]), axis=1)
        return out

def to_mne_raw(data: np.ndarray, ch_names: list, sfreq: float = SAMPLE_RATE):
    """
    data: (n_channels, n_samples)
    returns: mne.io.RawArray
    """
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw

def compute_psd_mne(data: np.ndarray, sfreq: float = SAMPLE_RATE, fmin=1.0, fmax=50.0, n_fft=None):
    """
    data: (n_channels, n_samples)
    returns freqs, psd (n_channels, n_freqs)
    Uses mne.time_frequency.psd_array_welch
    """
    if n_fft is None:
        n_fft = int(sfreq * 2)  # 2-second window default
    psd_all = []
    for ch in range(data.shape[0]):
        psd, freqs = psd_array_welch(data[ch, :], sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft, verbose=False)
        psd_all.append(psd)
    return freqs, np.vstack(psd_all)

def integrate_band(freqs: np.ndarray, psd: np.ndarray, low: float, high: float):
    """
    psd: shape (n_freqs,) or (n_channels, n_freqs)
    returns integrated power (scalar or array per channel)
    """
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        if psd.ndim == 1:
            return 0.0
        else:
            return np.zeros(psd.shape[0])
    if psd.ndim == 1:
        return float(np.trapz(psd[mask], freqs[mask]))
    else:
        return np.trapz(psd[:, mask], freqs[mask], axis=1)

# convenience wrapper
def bandpower_from_raw_block(block, ch_names, fmin, fmax, sfreq=SAMPLE_RATE, n_fft=None):
    freqs, psd = compute_psd_mne(block, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
    bp = integrate_band(freqs, psd, fmin, fmax)  # per-channel
    return freqs, psd, bp
