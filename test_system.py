#!/usr/bin/env python3
"""
End-to-end test of the ML communication system.

Usage:
    python test_system.py                        # AWGN, 10 dB
    python test_system.py --snr 5                # noisy
    python test_system.py --snr 15 --channel rayleigh
    python test_system.py --snr 10 --channel composite
    python test_system.py --all                  # test all channel types

Channel options: awgn | phase | freq | rayleigh | composite
"""

import numpy as np
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from models.autoencoder import CUSTOM_OBJECTS
from utils.channel_models import (
    awgn_channel,
    phase_shift_channel,
    frequency_offset_channel,
    rayleigh_fading_channel,
    composite_channel,
)
from utils.metrics import calculate_ber, bytes_to_bits


# ─────────────────────────────────────────────
# Channel selector
# ─────────────────────────────────────────────

CHANNEL_FN = {
    'awgn':      lambda sig, snr: awgn_channel(sig, snr),
    'phase':     lambda sig, snr: phase_shift_channel(sig, snr),
    'freq':      lambda sig, snr: frequency_offset_channel(sig, snr),
    'rayleigh':  lambda sig, snr: rayleigh_fading_channel(sig, snr),
    'composite': lambda sig, snr: composite_channel(sig, snr),
}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def bytes_to_bits_local(byte_val, k=8):
    return [int(b) for b in format(byte_val, f'0{k}b')]

def bits_to_byte(bits):
    return int(''.join(map(str, np.round(bits).astype(int))), 2)


def find_model_paths(base_dir):
    """Locate encoder and decoder model files (robust first, legacy fallback)."""
    saved = os.path.join(base_dir, "models", "saved_models")

    enc_candidates = [
        os.path.join(saved, "robust_encoder.keras"),
        os.path.join(saved, "encoder.keras"),
    ]
    dec_candidates = [
        os.path.join(saved, "robust_decoder.keras"),
        os.path.join(saved, "decoder.keras"),
        os.path.join(saved, "robust_autoencoder.keras"),
        os.path.join(saved, "autoencoder_final.keras"),
    ]

    encoder_path = next((p for p in enc_candidates if os.path.exists(p)), None)
    decoder_path = next((p for p in dec_candidates if os.path.exists(p)), None)
    return encoder_path, decoder_path


# ─────────────────────────────────────────────
# Core test function
# ─────────────────────────────────────────────

def test_ml_comm_system(snr_db=10.0, channel_mode='awgn', num_messages=16, k=8, n=4):
    """
    Run end-to-end encoder → channel → decoder test.

    Returns:
        dict with keys: accuracy, ber, correct, total
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path, decoder_path = find_model_paths(base_dir)

    if not encoder_path:
        print("ERROR: No encoder model found. Train first:\n  cd models && python train.py")
        return None

    channel_label = channel_mode.upper()
    print("=" * 70)
    print(f"ML Communication System — Test ({channel_label} channel, SNR={snr_db} dB)")
    print("=" * 70)
    print(f"  Encoder: {encoder_path}")
    print(f"  Decoder: {decoder_path}")
    print("=" * 70)

    # Load encoder
    encoder = tf.keras.models.load_model(
        encoder_path, compile=False, custom_objects=CUSTOM_OBJECTS
    )

    # Load decoder
    if decoder_path and os.path.exists(decoder_path):
        decoder = tf.keras.models.load_model(
            decoder_path, compile=False, custom_objects=CUSTOM_OBJECTS
        )
        # Detect model type by input dtype
        inp_dtype = str(decoder.input.dtype)
        decoder_takes_complex = 'complex' in inp_dtype
    else:
        decoder = None
        decoder_takes_complex = False

    if decoder is None:
        print("ERROR: No decoder model found.")
        return None

    # Generate test messages
    test_bytes_list = [i * (255 // (num_messages - 1)) for i in range(num_messages)]
    test_bytes_list = list(set(test_bytes_list))[:num_messages]  # unique values

    print(f"\nTesting {len(test_bytes_list)} messages ...")
    print(f"Channel: {channel_mode} | SNR: {snr_db} dB\n")

    channel_fn = CHANNEL_FN.get(channel_mode, CHANNEL_FN['awgn'])

    correct = 0
    total   = len(test_bytes_list)
    all_orig_bits = []
    all_dec_bits  = []

    print(f"{'Idx':<5} {'Orig':<8} {'Decoded':<10} {'Match'}")
    print("─" * 38)

    for idx, orig_byte in enumerate(test_bytes_list):
        # Convert byte → bits
        bits = np.array(bytes_to_bits_local(orig_byte, k), dtype=np.float32).reshape(1, k)

        # Encode
        enc_syms = encoder.predict(bits, verbose=0)[0]  # shape (n,) complex

        # Apply channel
        noisy_syms = channel_fn(enc_syms, snr_db)

        # Decode
        if decoder_takes_complex:
            noisy_in = noisy_syms.astype(np.complex64).reshape(1, n)
        else:
            # Decoder expects real (from autoencoder format): split IQ
            noisy_in = np.concatenate([
                np.real(noisy_syms), np.imag(noisy_syms)
            ]).astype(np.float32).reshape(1, -1)

        dec_bits = decoder.predict(noisy_in, verbose=0)[0]
        dec_byte  = bits_to_byte(dec_bits)

        match = "✓" if dec_byte == orig_byte else "✗"
        if dec_byte == orig_byte:
            correct += 1

        print(f"{idx:<5} {orig_byte:<8} {dec_byte:<10} {match}")

        all_orig_bits.append(bits_to_byte(bits[0]))
        all_dec_bits.append(dec_byte)

    print("─" * 38)
    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")

    # BER
    orig_bits_arr = np.array([bytes_to_bits_local(b, k) for b in all_orig_bits], dtype=float)
    dec_bits_arr  = np.array([bytes_to_bits_local(b, k) for b in all_dec_bits],  dtype=float)
    ber = calculate_ber(orig_bits_arr, dec_bits_arr)
    print(f"BER:      {ber:.6f}")
    print("=" * 70)

    return dict(accuracy=accuracy, ber=ber, correct=correct, total=total)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test ML communication system')
    parser.add_argument('--snr',     type=float, default=10.0,
                        help='SNR in dB (default: 10)')
    parser.add_argument('--channel', type=str,   default='awgn',
                        choices=list(CHANNEL_FN), metavar='MODE',
                        help='Channel type: awgn|phase|freq|rayleigh|composite')
    parser.add_argument('--messages',type=int,   default=16,
                        help='Number of test messages (default: 16)')
    parser.add_argument('--all',     action='store_true',
                        help='Test all channel types')
    args = parser.parse_args()

    if args.all:
        print("\n" + "=" * 70)
        print("TESTING ALL CHANNEL TYPES")
        print("=" * 70)
        summary = {}
        for ch in CHANNEL_FN:
            result = test_ml_comm_system(
                snr_db=args.snr, channel_mode=ch, num_messages=args.messages
            )
            if result:
                summary[ch] = result

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"{'Channel':<12} {'Accuracy':<12} {'BER'}")
        print("─" * 38)
        for ch, r in summary.items():
            print(f"{ch:<12} {r['accuracy']:.1f}%       {r['ber']:.6f}")
        print("=" * 70)
    else:
        result = test_ml_comm_system(
            snr_db=args.snr, channel_mode=args.channel, num_messages=args.messages
        )
        if result and result['accuracy'] >= 80:
            print("\n✓ Test PASSED")
            sys.exit(0)
        else:
            print("\n✗ Test FAILED (accuracy < 80%) — retrain or increase SNR")
            sys.exit(1)
