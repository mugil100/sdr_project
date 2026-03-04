"""
Simple script to trace and save the original k=4 AWGN autoencoder.
"""

import numpy as np
import tensorflow as tf
import os
import argparse

from autoencoder import create_autoencoder, compile_model

def generate_training_data(num_samples, k=4):
    return np.random.randint(0, 2, size=(num_samples, k)).astype(np.float32)

def train(k=4, n=4, snr_db=10.0, epochs=30, batch_size=256, samples=100000, 
          use_phase_offset=False, use_cfo=False, max_cfo=0.05):
    save_path = 'saved_models'
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Training Autoencoder (k={k}, n={n}, SNR={snr_db}dB, PO={use_phase_offset}, CFO={use_cfo})")
    train_data = generate_training_data(samples, k)
    val_data = generate_training_data(10000, k)
    
    autoencoder, encoder, decoder = create_autoencoder(
        k=k, n=n, snr_db=snr_db, 
        use_phase_offset=use_phase_offset, 
        use_cfo=use_cfo, max_cfo=max_cfo
    )
    autoencoder = compile_model(autoencoder)
    
    autoencoder.fit(
        train_data, train_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data, val_data),
        verbose=1
    )
    
    suffix = ""
    if use_phase_offset and use_cfo:
        suffix = "_po_cfo"
    elif use_phase_offset:
        suffix = "_po"
    elif use_cfo:
        suffix = "_cfo"

    autoencoder.save(os.path.join(save_path, f'autoencoder_final{suffix}.keras'))
    encoder.save(os.path.join(save_path, f'encoder{suffix}.keras'))
    decoder.save(os.path.join(save_path, f'decoder{suffix}.keras'))
    print("Models saved to models/saved_models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--n', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--use_phase_offset', action='store_true', help='Enable random phase offset during training')
    parser.add_argument('--use_cfo', action='store_true', help='Enable carrier frequency offset during training')
    parser.add_argument('--max_cfo', type=float, default=0.05, help='Maximum normalized frequency offset')
    args = parser.parse_args()
    
    train(
        k=args.k, n=args.n, epochs=args.epochs,
        use_phase_offset=args.use_phase_offset,
        use_cfo=args.use_cfo, max_cfo=args.max_cfo
    )
