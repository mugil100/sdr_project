import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys

# Ensure custom modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.autoencoder import CUSTOM_OBJECTS
from utils.metrics import calculate_ber
from utils.channel_models import composite_channel

def load_models():
    """Load both baseline and Rayleigh robust models."""
    try:
        baseline_model = tf.keras.models.load_model(
            'models/saved_models/encoder.keras', 
            custom_objects=CUSTOM_OBJECTS
        )
        robust_model = tf.keras.models.load_model(
            'models/saved_models/encoder_fading.keras', 
            custom_objects=CUSTOM_OBJECTS
        )
        
        baseline_decoder = tf.keras.models.load_model(
            'models/saved_models/decoder.keras',
            custom_objects=CUSTOM_OBJECTS
        )
        robust_decoder = tf.keras.models.load_model(
            'models/saved_models/decoder_fading.keras',
            custom_objects=CUSTOM_OBJECTS
        )
        return (baseline_model, baseline_decoder), (robust_model, robust_decoder)
    except Exception as e:
        print(f"Error loading models: {e}")
        return (None, None), (None, None)

def evaluate_ber(encoder, decoder, snr_db, num_bytes=10000, 
                 use_po=False, use_cfo=False, use_fading=False, k=8):
    """Evaluate BER for a given model at a specific SNR under specific channel conditions."""
    # Generate random bits (k bits per symbol transmission)
    bits = np.random.randint(0, 2, size=(num_bytes, k)).astype(np.float32)
    
    # Encode
    symbols = encoder.predict(bits, verbose=0)
    
    # Pass through dynamic numpy composite channel
    received = composite_channel(
        symbols, 
        snr_db=snr_db, 
        phase_deg=np.pi/4 if use_po else 0.0,
        freq_offset_normalized=0.05 if use_cfo else 0.0,
        fading=use_fading
    )
    
    # Decode
    decoded_probs = decoder.predict(received, verbose=0)
    decoded_bits = np.round(decoded_probs).astype(int)
    
    # Compare
    ber = calculate_ber(bits.astype(int), decoded_bits)
    return ber

def main():
    print("Loading Baseline and Fading Robust Models...")
    (base_enc, base_dec), (rob_enc, rob_dec) = load_models()
    
    if base_enc is None or rob_enc is None:
        print("Required models are missing. Train them first!")
        print("  python models/train.py --k 8")
        print("  python models/train.py --k 8 --use_fading")
        sys.exit(1)
        
    snrs = np.arange(0, 22, 2)
    
    print("\nEvaluating Scenario: Rayleigh Flat Fading")
    base_ber, rob_ber = [], []
    for snr in snrs:
        b = evaluate_ber(base_enc, base_dec, snr, use_fading=True)
        r = evaluate_ber(rob_enc, rob_dec, snr, use_fading=True)
        base_ber.append(b)
        rob_ber.append(r)
        print(f"  SNR {snr:2d}dB -> Baseline BER: {b:.4f} | Robust BER: {r:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(10, 7))
    
    plt.semilogy(snrs, base_ber, 'bo-', label='Baseline (AWGN trained)', linewidth=2, markersize=8)
    plt.semilogy(snrs, rob_ber, 'r^-', label='Fading Robust', linewidth=2, markersize=8)
        
    plt.title('BER vs SNR under Rayleigh Flat Fading')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    # Create images directory if not exists
    os.makedirs('images', exist_ok=True)
    out_path = 'images/fading_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {out_path}")
    
if __name__ == "__main__":
    main()
