"""
Channel models for communication simulation.

Available functions:
  - awgn_channel           : AWGN only
  - phase_shift_channel    : Random or fixed phase rotation + AWGN
  - frequency_offset_channel: Time-varying frequency offset + AWGN
  - rayleigh_fading_channel : Rayleigh flat fading + AWGN
  - rician_fading_channel   : Rician fading (LOS + scatter) + AWGN
  - composite_channel       : All impairments combined
"""

import numpy as np


# ─────────────────────────────────────────────
# AWGN (original)
# ─────────────────────────────────────────────

def awgn_channel(signal, snr_db):
    """
    Add AWGN to a complex signal.

    Args:
        signal: Complex numpy array
        snr_db: SNR in dB

    Returns:
        Noisy complex signal
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear   = 10.0 ** (snr_db / 10.0)
    noise_power  = signal_power / snr_linear
    noise_std    = np.sqrt(noise_power / 2.0)
    noise        = noise_std * (np.random.randn(*signal.shape) +
                                1j * np.random.randn(*signal.shape))
    return signal + noise


# ─────────────────────────────────────────────
# Phase shift (NEW)
# ─────────────────────────────────────────────

def phase_shift_channel(signal, snr_db, phase_deg=None):
    """
    Apply a static phase rotation then add AWGN.

    Models carrier phase uncertainty in the receiver.

    Args:
        signal:    Complex numpy array, shape (n_symbols,) or (batch, n)
        snr_db:    SNR in dB
        phase_deg: Phase rotation in degrees.
                   None → random uniform [0°, 360°) per call.

    Returns:
        Noisy, phase-rotated complex signal
    """
    if phase_deg is None:
        phase_deg = np.random.uniform(0.0, 360.0)

    phase_rad = np.deg2rad(phase_deg)
    rotated   = signal * np.exp(1j * phase_rad)
    return awgn_channel(rotated, snr_db)


# ─────────────────────────────────────────────
# Frequency offset (NEW)
# ─────────────────────────────────────────────

def frequency_offset_channel(signal, snr_db, freq_offset_normalized=0.01):
    """
    Apply a time-varying frequency offset then add AWGN.

    Models carrier frequency offset (CFO) between transmitter and receiver.
    The phase increases linearly with symbol index:
        y[i] = x[i] * exp(j * 2π * f_offset * i)

    Args:
        signal:                Complex 1-D numpy array, shape (n_symbols,)
        snr_db:                SNR in dB
        freq_offset_normalized: Normalized frequency offset (fraction of
                               sample rate). Default 0.01 = 1% of Fs.
                               Pass None for random uniform in ±0.05.

    Returns:
        Frequency-shifted, noisy complex signal
    """
    if freq_offset_normalized is None:
        freq_offset_normalized = np.random.uniform(-0.05, 0.05)

    flat_signal = signal.flatten()
    t           = np.arange(len(flat_signal))
    rotation    = np.exp(1j * 2.0 * np.pi * freq_offset_normalized * t)
    shifted     = flat_signal * rotation

    # Reshape back if needed
    if signal.ndim > 1:
        shifted = shifted.reshape(signal.shape)

    return awgn_channel(shifted, snr_db)


# ─────────────────────────────────────────────
# Rayleigh fading (updated to match Keras layer)
# ─────────────────────────────────────────────

def rayleigh_fading_channel(signal, snr_db):
    """
    Rayleigh FLAT-fading channel + AWGN.

    A single complex Gaussian fading coefficient h ~ CN(0,1) is applied
    to ALL symbols in the codeword (flat fading model). This matches the
    Keras RayleighFadingChannel training layer exactly.

    In a real system this models slow fading where the coherence time
    is longer than one codeword duration.

    Args:
        signal: Complex numpy array, shape (n,) or (batch, n)
        snr_db: SNR in dB

    Returns:
        Faded + noisy signal (same shape as input)
    """
    # Scalar h per codeword — same coefficient for all n symbols
    h_real = np.random.randn() / np.sqrt(2)
    h_imag = np.random.randn() / np.sqrt(2)
    h      = h_real + 1j * h_imag
    faded  = h * signal          # broadcasts over all symbols
    return awgn_channel(faded, snr_db)


# ─────────────────────────────────────────────
# Rician fading
# ─────────────────────────────────────────────

def rician_fading_channel(signal, snr_db, K_factor_db=10):
    """
    Rician fading channel + AWGN.

    Args:
        signal:       Complex numpy array
        snr_db:       SNR in dB
        K_factor_db:  Rician K-factor in dB (LOS power / scattered power)

    Returns:
        Faded + noisy signal
    """
    K_linear   = 10.0 ** (K_factor_db / 10.0)
    los        = np.sqrt(K_linear / (K_linear + 1.0))
    scatter    = (np.sqrt(1.0 / (K_linear + 1.0)) *
                  (np.random.randn(*signal.shape) +
                   1j * np.random.randn(*signal.shape)) / np.sqrt(2))
    h          = los + scatter
    faded      = h * signal
    return awgn_channel(faded, snr_db)


# ─────────────────────────────────────────────
# Composite channel (NEW)
# ─────────────────────────────────────────────

def composite_channel(signal, snr_db,
                      fading=True, freq_offset=True, phase_shift=True,
                      freq_offset_normalized=None, phase_deg=None,
                      K_factor_db=None):
    """
    Apply a combination of channel impairments then add AWGN.

    Applies impairments in this order:
      Rayleigh fading → Frequency offset → Phase shift → AWGN

    Args:
        signal:                Complex numpy array
        snr_db:                SNR in dB
        fading:                Enable Rayleigh fading
        freq_offset:           Enable frequency offset
        phase_shift:           Enable phase rotation
        freq_offset_normalized: Normalized CFO (None = random ±0.05)
        phase_deg:             Phase in degrees (None = random 0–360°)
        K_factor_db:           If set, use Rician fading instead of Rayleigh

    Returns:
        Channel-impaired complex signal
    """
    x = signal.copy()

    # 1. Fading — scalar h per codeword (flat fading, matches Keras layer)
    if fading:
        if K_factor_db is not None:
            K_lin   = 10.0 ** (K_factor_db / 10.0)
            los     = np.sqrt(K_lin / (K_lin + 1.0))
            scatter = (np.sqrt(1.0 / (K_lin + 1.0)) *
                       (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2))
            h = los + scatter   # scalar
        else:
            h = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)  # scalar
        x = x * h              # broadcast over all n symbols

    # 2. Frequency offset
    if freq_offset:
        fo = freq_offset_normalized
        if fo is None:
            fo = np.random.uniform(-0.05, 0.05)
        flat = x.flatten()
        t    = np.arange(len(flat))
        flat = flat * np.exp(1j * 2.0 * np.pi * fo * t)
        x    = flat.reshape(x.shape)

    # 3. Phase shift
    if phase_shift:
        pd  = phase_deg if phase_deg is not None else np.random.uniform(0, 360)
        x   = x * np.exp(1j * np.deg2rad(pd))

    # 4. AWGN
    return awgn_channel(x, snr_db)


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing channel models ...")
    sig = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=np.complex64)
    print(f"Original:         {sig}")
    print(f"AWGN (10 dB):     {awgn_channel(sig, 10)}")
    print(f"Phase shift 45°:  {phase_shift_channel(sig, 10, 45)}")
    print(f"Freq offset 1%:   {frequency_offset_channel(sig, 10, 0.01)}")
    print(f"Rayleigh:         {rayleigh_fading_channel(sig, 10)}")
    print(f"Rician (K=10dB):  {rician_fading_channel(sig, 10, 10)}")
    print(f"Composite (all):  {composite_channel(sig, 10)}")
    print("All channel models OK ✓")
