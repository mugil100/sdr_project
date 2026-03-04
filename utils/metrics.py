"""
Performance metrics for communication systems
"""

import numpy as np


def calculate_ber(original_bits, decoded_bits, threshold=0.5):
    """
    Calculate Bit Error Rate (BER)
    
    Args:
        original_bits: Original binary data
        decoded_bits: Decoded binary data (may be soft values)
        threshold: Decision threshold for soft values
    
    Returns:
        ber: Bit error rate
    """
    original = np.round(original_bits).astype(int).flatten()
    decoded = (decoded_bits > threshold).astype(int).flatten()
    
    errors = np.sum(original != decoded)
    total_bits = len(original)
    
    return errors / total_bits if total_bits > 0 else 0


def calculate_ser(original_symbols, decoded_symbols):
    """
    Calculate Symbol Error Rate (SER)
    
    Args:
        original_symbols: Original symbols
        decoded_symbols: Decoded symbols
    
    Returns:
        ser: Symbol error rate
    """
    errors = np.sum(original_symbols != decoded_symbols)
    total_symbols = len(original_symbols)
    
    return errors / total_symbols if total_symbols > 0 else 0


def calculate_evm(transmitted, received):
    """
    Calculate Error Vector Magnitude (EVM)
    
    Args:
        transmitted: Transmitted complex symbols
        received: Received complex symbols
    
    Returns:
        evm_percent: EVM in percentage
    """
    error = received - transmitted
    error_power = np.mean(np.abs(error) ** 2)
    signal_power = np.mean(np.abs(transmitted) ** 2)
    
    evm = np.sqrt(error_power / signal_power)
    return evm * 100  # Return as percentage


def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio (SNR)
    
    Args:
        signal: Clean signal
        noise: Noise component
    
    Returns:
        snr_db: SNR in dB
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_linear = signal_power / noise_power
    return 10 * np.log10(snr_linear)


def bits_to_bytes(bits):
    """
    Convert bit array to byte values (0-255)
    
    Args:
        bits: Array of bits, shape (n, 8)
    
    Returns:
        bytes: Array of byte values (0-255)
    """
    if bits.ndim == 1:
        bits = bits.reshape(-1, 8)
    
    bytes_array = []
    for row in bits:
        byte_val = int(''.join(map(str, row.astype(int))), 2)
        bytes_array.append(byte_val)
    
    return np.array(bytes_array)


def bytes_to_bits(bytes_array, k=8):
    """
    Convert byte values to bit array
    
    Args:
        bytes_array: Array of byte values (0-255)
        k: Number of bits per byte
    
    Returns:
        bits: Array of bits, shape (n, k)
    """
    bits = []
    for byte_val in bytes_array:
        bit_string = format(int(byte_val), f'0{k}b')
        bits.append([int(b) for b in bit_string])
    
    return np.array(bits, dtype=np.float32)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")
    
    # BER test
    original = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    decoded = np.array([1, 0, 1, 0, 0, 1, 0, 0])  # 1 error
    ber = calculate_ber(original, decoded)
    print(f"BER: {ber} (expected: 0.125)")
    
    # Conversion test
    test_bytes = np.array([23, 45, 255])
    bits = bytes_to_bits(test_bytes)
    print(f"\nBytes to bits:")
    print(f"Bytes: {test_bytes}")
    print(f"Bits:\n{bits}")
    
    recovered_bytes = bits_to_bytes(bits)
    print(f"Recovered bytes: {recovered_bytes}")
    print(f"Match: {np.array_equal(test_bytes, recovered_bytes)}")
