"""
ML Encoder Block for GNU Radio
Converts byte stream to complex IQ samples using trained neural network
"""

import numpy as np
import os
import sys

# Ensure parent dir is on path so CUSTOM_OBJECTS can be imported
_BLOCK_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_BLOCK_DIR)
for _p in [_PROJECT_DIR, os.path.join(_PROJECT_DIR, 'models')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tensorflow as tf

try:
    from autoencoder import CUSTOM_OBJECTS
except ImportError:
    CUSTOM_OBJECTS = {}
    print('[ML Encoder] Warning: CUSTOM_OBJECTS not found, loading without custom layers')

try:
    from gnuradio import gr
    _HAS_GR = True
except ImportError:
    _HAS_GR = False


def _resolve_model_path(model_path):
    """
    Given ANY model path, return the best available encoder .keras file.
    Prefers robust_encoder.keras over the legacy encoder.keras.
    """
    base_dir = os.path.join(_PROJECT_DIR, 'models', 'saved_models')
    candidates = [
        os.path.join(base_dir, 'encoder_po_cfo.keras'),
        os.path.join(base_dir, 'robust_encoder.keras'),
        os.path.join(base_dir, 'encoder.keras'),
        model_path,  # whatever the user passed
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return model_path  # fallback (will raise later)


class ml_encoder(gr.sync_block if _HAS_GR else object):
    """
    ML-based encoder block.
    Input:  byte stream  (uint8,     0-255)
    Output: complex IQ   (complex64, 4 symbols per byte)
    """

    def __init__(self, model_path='', k=8, n=4):
        """
        Args:
            model_path: Path to encoder .keras file (auto-resolved to
                        robust_encoder.keras if the path is wrong/old)
            k:          Bits per input byte (default 8)
            n:          Complex symbols per byte (default 4)
        """
        if _HAS_GR:
            gr.sync_block.__init__(
                self,
                name='ml_encoder',
                in_sig=[np.uint8],
                out_sig=[np.complex64]
            )

        self.k = k
        self.n = n

        # Auto-resolve: prefer robust_encoder.keras
        resolved_path = _resolve_model_path(model_path)
        print(f'[ML Encoder] Loading: {resolved_path}')

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f'[ML Encoder] No encoder model found.\n'
                f'  Tried: {resolved_path}\n'
                f'  Train first: cd models && python train.py'
            )

        # Load with custom objects — required for PowerNormalization, ToComplex
        self.encoder = tf.keras.models.load_model(
            resolved_path, compile=False, custom_objects=CUSTOM_OBJECTS
        )
        print(f'[ML Encoder] Loaded  ({k} bits → {n} complex symbols)')

        if _HAS_GR:
            self.set_output_multiple(n)

    
    def work(self, input_items, output_items):
        """
        Process input bytes and generate complex symbols
        """
        in_bytes = input_items[0]
        out_symbols = output_items[0]
        
        # Number of complete bytes to process
        num_bytes = len(in_bytes)
        num_output_samples = len(out_symbols)
        max_bytes = num_output_samples // self.n
        
        num_to_process = min(num_bytes, max_bytes)
        
        if num_to_process == 0:
            return 0
        
        # Convert bytes to bits
        bits_batch = []
        for i in range(num_to_process):
            byte_val = int(in_bytes[i])
            # Convert to binary (8 bits)
            bits = [int(b) for b in format(byte_val, '08b')]
            bits_batch.append(bits)
        
        bits_array = np.array(bits_batch, dtype=np.float32)
        
        # Encode using neural network
        encoded_symbols = self.encoder.predict(bits_array, verbose=0)
        
        # Flatten to output stream
        out_symbols[:num_to_process * self.n] = encoded_symbols.flatten()
        
        return num_to_process * self.n


# For standalone testing
if __name__ == "__main__":
    import sys
    
    # Test the encoder block
    print("Testing ML Encoder Block")
    print("=" * 60)
    
    model_path = ""
    
    try:
        # Create encoder block
        encoder_block = ml_encoder(model_path=model_path, k=8, n=4)
        
        # Test with sample data
        test_bytes = np.array([23, 45, 100, 200, 255], dtype=np.uint8)
        output_buffer = np.zeros(5 * 4, dtype=np.complex64)
        
        # Simulate work function
        num_produced = encoder_block.work([test_bytes], [output_buffer])
        
        print(f"Input bytes: {test_bytes}")
        print(f"Produced {num_produced} complex symbols")
        print(f"Output symbols shape: {output_buffer[:num_produced].shape}")
        print(f"First few symbols:")
        for i in range(min(4, num_produced)):
            print(f"  Symbol {i}: {output_buffer[i].real:.4f} + {output_buffer[i].imag:.4f}j")
        
        print("\n" + "=" * 60)
        print("Encoder block test PASSED ✓")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
