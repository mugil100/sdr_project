#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Python GNU Radio Flowgraph for ML Communication System
No GUI required - can run headless
"""

import numpy as np
from gnuradio import gr, blocks, channels
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_blocks.encoder_block import ml_encoder
from ml_blocks.decoder_block import ml_decoder


class ml_comm_flowgraph(gr.top_block):
    """
    ML-based communication system flowgraph
    """
    
    def __init__(self, snr_db=10.0, num_samples=100):
        gr.top_block.__init__(self, "ML Communication System")
        
        print("=" * 70)
        print("ML Communication System - GNU Radio Flowgraph")
        print("=" * 70)
        print(f"SNR: {snr_db} dB")
        print(f"Processing {num_samples} messages")
        print("=" * 70)
        
        # Parameters
        self.samp_rate = 32000
        self.snr_db = snr_db
        
        # Model paths
        encoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "saved_models", "encoder.keras"
        )
        decoder_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "saved_models", "autoencoder_final.keras"
        )
        
        # Check models exist
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
        
        # Generate test data
        test_data = list(np.random.randint(0, 256, num_samples, dtype=np.uint8))
        
        ##################################################
        # Blocks
        ##################################################
        
        # Source: Vector of bytes to transmit
        self.source = blocks.vector_source_b(test_data, False)
        
        # Throttle to control flow rate
        self.throttle = blocks.throttle(gr.sizeof_char, self.samp_rate, True)
        
        # ML Encoder
        self.ml_encoder = ml_encoder(model_path=encoder_path, k=8, n=4)
        
        # Channel: AWGN
        noise_voltage = 10**(-self.snr_db/20.0)
        self.channel = channels.channel_model(
            noise_voltage=noise_voltage,
            frequency_offset=0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=42
        )
        
        # ML Decoder  
        self.ml_decoder = ml_decoder(model_path=decoder_path, k=8, n=4)
        
        # Sinks
        self.input_sink = blocks.vector_sink_b()
        self.output_sink = blocks.vector_sink_b()
        
        # File sinks for inspection
        self.file_input = blocks.file_sink(gr.sizeof_char, "transmitted_bytes.dat", False)
        self.file_output = blocks.file_sink(gr.sizeof_char, "received_bytes.dat", False)
        
        ##################################################
        # Connections
        ##################################################
        
        # Source → Throttle → Encoder → Channel → Decoder → Sink
        self.connect((self.source, 0), (self.throttle, 0))
        self.connect((self.throttle, 0), (self.ml_encoder, 0))
        self.connect((self.ml_encoder, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.ml_decoder, 0))
        
        # Connect sinks
        self.connect((self.throttle, 0), (self.input_sink, 0))
        self.connect((self.throttle, 0), (self.file_input, 0))
        self.connect((self.ml_decoder, 0), (self.output_sink, 0))
        self.connect((self.ml_decoder, 0), (self.file_output, 0))
    
    def get_results(self):
        """Get transmitted and received data for comparison"""
        input_data = np.array(self.input_sink.data(), dtype=np.uint8)
        output_data = np.array(self.output_sink.data(), dtype=np.uint8)
        return input_data, output_data


def main():
    """Run the flowgraph and display results"""
    
    import argparse
    parser = argparse.ArgumentParser(description='ML Communication System Flowgraph')
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB')
    parser.add_argument('--samples', type=int, default=50, help='Number of messages')
    args = parser.parse_args()
    
    try:
        # Create and run flowgraph
        print("\nCreating flowgraph...")
        tb = ml_comm_flowgraph(snr_db=args.snr, num_samples=args.samples)
        
        print("Running flowgraph...")
        tb.start()
        
        # Wait for completion
        tb.wait()
        
        # Get results
        print("\nProcessing results...")
        input_data, output_data = tb.get_results()
        
        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        # Compare first 20 messages
        num_to_show = min(20, len(input_data), len(output_data))
        
        print(f"\n{'Index':<8} {'Transmitted':<15} {'Received':<15} {'Match':<8}")
        print("-" * 55)
        
        correct = 0
        total = min(len(input_data), len(output_data))
        
        for i in range(total):
            if i < num_to_show:
                match = "✓" if input_data[i] == output_data[i] else "✗"
                print(f"{i:<8} {input_data[i]:<15} {output_data[i]:<15} {match:<8}")
            
            if input_data[i] == output_data[i]:
                correct += 1
        
        if total > num_to_show:
            print(f"... ({total - num_to_show} more messages)")
        
        print("-" * 55)
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        # Calculate BER
        from utils.metrics import bytes_to_bits, calculate_ber
        input_bits = bytes_to_bits(input_data[:total])
        output_bits = bytes_to_bits(output_data[:total])
        ber = calculate_ber(input_bits, output_bits)
        
        print(f"Bit Error Rate (BER): {ber:.6f}")
        
        print("\nData files saved:")
        print("  - transmitted_bytes.dat")
        print("  - received_bytes.dat")
        
        print("=" * 70)
        print("✓ Flowgraph completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
