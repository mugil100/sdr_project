import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys

from models.autoencoder import CUSTOM_OBJECTS
from utils.metrics import calculate_ber
from utils.channel_models import composite_channel

def load_models():
    """Load both baseline and robust models."""
    try:
        baseline_enc = keras.models.load_model('models/saved_models/encoder.keras', custom_objects=CUSTOM_OBJECTS)
        baseline_dec = keras.models.load_model('models/saved_models/decoder.keras', custom_objects=CUSTOM_OBJECTS)
        
        robust_enc = keras.models.load_model('models/saved_models/encoder_po_cfo.keras', custom_objects=CUSTOM_OBJECTS)
        robust_dec = keras.models.load_model('models/saved_models/decoder_po_cfo.keras', custom_objects=CUSTOM_OBJECTS)
        
        return (baseline_enc, baseline_dec), (robust_enc, robust_dec)
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def evaluate_ber(encoder, decoder, snr_db, num_bytes=10000, 
                 use_po=False, use_cfo=False, k=8):
    """Evaluate BER for a given model at a specific SNR under specific channel conditions."""
    # Generate random bits (k bits per symbol transmission)
    bits = np.random.randint(0, 2, size=(num_bytes, k)).astype(np.float32)
    
    # Encode
    symbols = encoder.predict(bits, verbose=0)
    
    # Process through Channel
    # Convert phase_shift to degrees, and normalize CFO if enabled
    phase_deg = None if use_po else 0.0
    cfo = None if use_cfo else 0.0
    
    noisy_symbols = composite_channel(
        symbols, 
        snr_db=snr_db,
        fading=False, 
        freq_offset=use_cfo, 
        phase_shift=use_po,
        freq_offset_normalized=cfo,
        phase_deg=phase_deg
    )
    
    # Decode
    decoded_probs = decoder.predict(noisy_symbols, verbose=0)
    decoded_bits = (decoded_probs > 0.5).astype(np.float32)
    
    # Calculate BER
    return calculate_ber(bits, decoded_bits)

def run_comparison():
    baseline_models, robust_models = load_models()
    if not baseline_models:
        return
        
    baseline_enc, baseline_dec = baseline_models
    robust_enc,   robust_dec   = robust_models
    
    snrs = np.arange(0, 21, 2)
    
    # Scenarios:
    # 1. AWGN only
    # 2. Phase Offset only
    # 3. CFO only
    # 4. Phase Offset + CFO
    
    scenarios = [
        ("AWGN Only", False, False),
        ("Phase Offset (PO)", True, False),
        ("Carrier Frequency Offset (CFO)", False, True),
        ("PO + CFO combined", True, True)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, use_po, use_cfo) in enumerate(scenarios):
        print(f"\nEvaluating Scenario: {name}")
        ber_base = []
        ber_robust = []
        
        for snr in snrs:
            b_ber = evaluate_ber(baseline_enc, baseline_dec, snr, use_po=use_po, use_cfo=use_cfo)
            r_ber = evaluate_ber(robust_enc,   robust_dec,   snr, use_po=use_po, use_cfo=use_cfo)
            
            ber_base.append(b_ber)
            ber_robust.append(r_ber)
            print(f"  SNR {snr:2d}dB -> Baseline BER: {b_ber:.4f} | Robust BER: {r_ber:.4f}")
            
        ax = axes[i]
        ax.semilogy(snrs, ber_base, 'r-o', label='Baseline (AWGN trained)')
        ax.semilogy(snrs, ber_robust, 'b-o', label='Robust (PO/CFO trained)')
        ax.set_title(name)
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend()
        
    # Create images directory if not exists
    os.makedirs('images', exist_ok=True)
    out_path = 'images/po_cfo_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")

if __name__ == "__main__":
    run_comparison()
