# ML-Based Communication System in GNU Radio

## Overview
This project implements a deep learning autoencoder-based communication system in GNU Radio Companion, replacing traditional blocks like samplers, quantizers, and modulators with neural networks.

## Project Specifications
- **Data Type**: Single byte messages (0-255)
- **Channel Model**: AWGN (Additive White Gaussian Noise)
- **GNU Radio Version**: 3.10.12.0
- **Python Version**: 3.12.9
- **ML Framework**: TensorFlow/Keras

## Quick Start

**Option 1: Automated Setup (Recommended)**
```bash
python quickstart.py
```

**Option 2: Manual Setup**
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
cd models
python train.py

# 3. Test the system
cd ..
python test_system.py

# 4. Run GNU Radio flowgraph
cd flowgraphs
python ml_comm_flowgraph.py
```

**For detailed step-by-step guide, see: [GETTING_STARTED.md](GETTING_STARTED.md)**

## Architecture
- **Encoder (Transmitter)**: Neural network that maps 8 bits → 4 complex symbols
- **Channel**: AWGN noise simulation  
- **Decoder (Receiver)**: Neural network that maps 4 symbols → 8 bits

## How It Works

### Traditional vs ML Approach

**Traditional:**
```
Message (23) → Binary (00010111) → QPSK Modulator → [4 symbols] → Channel → QPSK Demodulator → Binary → Message
```

**Our ML Approach:**
```
Message (23) → Binary (00010111) → ML Encoder → [4 complex symbols] → Channel → ML Decoder → Binary → Message (23)
                                         ↑                                            ↑
                                   Learned optimal                              Learned optimal
                                   representation                               recovery strategy
```

### Advantages

✓ **End-to-end optimization**: Encoder and decoder trained together  
✓ **Adaptive**: Learns best strategy for given channel  
✓ **Performance**: Can outperform fixed modulation schemes  
✓ **No manual design**: No need to design constellation diagrams  
```

## How It Works

1. **Training Phase**:
   - Generate random byte values (0-255)
   - Train autoencoder end-to-end through AWGN channel
   - Learn optimal encoding/decoding strategy

2. **Deployment Phase**:
   - Load trained model into custom GNU Radio blocks
   - Encoder converts bytes to complex IQ samples
   - Decoder recovers bytes from noisy IQ samples

## Advantages Over Traditional Methods
- **End-to-end optimization**: Jointly optimized transmitter and receiver
- **Adaptability**: Can retrain for different channels
- **Performance**: Potentially lower bit error rates
