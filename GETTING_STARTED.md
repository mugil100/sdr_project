# Getting Started with ML-Based Communication System

This guide will walk you through setting up and running your ML-based communication system in GNU Radio.

## Prerequisites

✓ Python 3.12.9  
✓ GNU Radio 3.10.12.0  
✓ Basic understanding of communication systems

## Step-by-Step Setup

### Step 1: Install Python Dependencies

Open PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd "C:\Users\vasan\OneDrive\Desktop\SDR ASSIGNMENT"
pip install -r requirements.txt
```

This will install:
- TensorFlow (for neural networks)
- NumPy (for numerical operations)
- Matplotlib (for plotting)
- SciPy (for scientific computing)

**Expected time**: 2-5 minutes

---

### Step 2: Train the Autoencoder Model

The neural network needs to be trained before it can encode/decode messages.

```powershell
cd models
python train.py
```

**What happens:**
- Generates 100,000 random 8-bit messages (0-255)
- Trains encoder to map 8 bits → 4 complex symbols
- Trains decoder to recover 8 bits from 4 symbols
- Saves trained models to `models/saved_models/`
- Creates training history plots

**Training time**: 5-15 minutes (depending on your CPU/GPU)

**Output:**
```
=========================================================
ML-Based Communication System - Training
=========================================================
Configuration:
  - Bits per symbol (k): 8
  - Complex symbols (n): 4
  - Training SNR: 10 dB
  - Epochs: 100
  ...
Training complete!
=========================================================
```

**Verification**: Check that these files exist:
- `models/saved_models/encoder.keras`
- `models/saved_models/decoder.keras`
- `models/saved_models/autoencoder_final.keras`
- `models/saved_models/training_history.png`

---

### Step 3: Test the Trained Model

Verify the model works correctly:

```powershell
cd ..
python test_system.py
```

Or with custom SNR:
```powershell
python test_system.py --snr 15
```

**Expected output:**
```
=========================================================
RESULTS
=========================================================
Original     Decoded      Match   
-----------------------------------
23           23           ✓       
45           45           ✓       
100          100          ✓       
...
Accuracy: 8/8 (100.0%)
Bit Error Rate (BER): 0.000000
```

---

### Step 4: Run the GNU Radio Flowgraph

#### Option A: Python Flowgraph (Recommended)

```powershell
cd flowgraphs
python ml_comm_flowgraph.py
```

With custom parameters:
```powershell
python ml_comm_flowgraph.py --snr 5 --samples 100
```

**What it does:**
- Creates a complete GNU Radio flowgraph programmatically
- Generates random test messages (0-255)
- Encodes using ML encoder
- Passes through AWGN channel
- Decodes using ML decoder
- Compares transmitted vs received messages

#### Option B: GNU Radio Companion (GUI)

```powershell
gnuradio-companion
```

Then follow the instructions in `flowgraphs/README.md` to create a visual flowgraph.

---

### Step 5: Experiment!

Try different scenarios:

**Test at different SNR levels:**
```powershell
python test_system.py --snr 0   # Very noisy
python test_system.py --snr 5   # Noisy
python test_system.py --snr 10  # Moderate
python test_system.py --snr 20  # Clean
```

**Send specific messages:**

Edit `test_system.py` line 32:
```python
test_messages = [23, 45, 100, 200, 255, 0, 128, 42]
# Change to your messages:
test_messages = [65, 66, 67]  # 'ABC' in ASCII
```

**Visualize BER vs SNR:**

After training, check `models/saved_models/ber_vs_snr.png` to see how performance changes with noise.

---

## Understanding the System

### How It Works

1. **Input**: Byte value (0-255), e.g., `23`
   - Binary: `00010111`

2. **Encoder (Neural Network)**:
   - Converts 8 bits → 4 complex symbols
   - Example: `[0.5+0.3j, -0.2+0.8j, 0.7-0.1j, -0.4-0.6j]`
   - Power normalized for transmission

3. **Channel (AWGN)**:
   - Adds noise based on SNR
   - Simulates wireless transmission

4. **Decoder (Neural Network)**:
   - Receives noisy symbols
   - Recovers original 8 bits
   - Outputs byte value, e.g., `23`

### Why ML Instead of Traditional Methods?

**Traditional approach:**
```
Bits → Modulator (QPSK/QAM) → Channel → Demodulator → Bits
        ↑ Hand-designed                    ↑ Hand-designed
```

**ML approach:**
```
Bits → ML Encoder → Channel → ML Decoder → Bits
        ↑ Learned from data!      ↑ Learned from data!
```

**Advantages:**
- ✓ End-to-end optimization
- ✓ Adapts to channel characteristics  
- ✓ Can outperform fixed modulation schemes
- ✓ No need to design constellation manually

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```powershell
pip install tensorflow
```

### "FileNotFoundError: encoder model not found"

**Solution:**
You need to train the model first:
```powershell
cd models
python train.py
```

### "ImportError: No module named 'gnuradio'"

**Solution:**
Ensure GNU Radio is installed and on your Python path. Try:
```powershell
python -c "import gnuradio; print(gnuradio.__version__)"
```

### Low accuracy / High BER

**Possible causes:**
1. **Model not trained enough**: Increase epochs in `train.py`
2. **SNR too low**: Test with higher SNR (e.g., 15-20 dB)
3. **Model/environment mismatch**: Retrain with current SNR

**Solution:**
```powershell
cd models
python train.py  # Retrain with more epochs or different SNR
```

---

## Next Steps

### 1. Improve Performance
- Train with more data: Edit `train.py`, increase `num_samples`
- Use deeper networks: Modify `autoencoder.py`, add more layers
- Train across SNR range: Use random SNR during training

### 2. Try Different Channels
- Rayleigh fading: Edit `flowgraphs/ml_comm_flowgraph.py`, replace AWGN channel
- Rician fading: Use `utils/channel_models.py`

### 3. Real Hardware Testing
- Connect USRP or RTL-SDR
- Replace channel model with hardware source/sink
- Test over-the-air transmission

### 4. Extend Functionality
- Variable length messages
- Error correction coding
- Multiple users (MIMO)
- Adaptive modulation

---

## File Structure Reference

```
SDR ASSIGNMENT/
├── models/
│   ├── autoencoder.py          # Neural network architecture
│   ├── train.py                # Training script
│   └── saved_models/           # Trained model files
│       ├── encoder.keras
│       ├── decoder.keras
│       └── autoencoder_final.keras
│
├── ml_blocks/
│   ├── encoder_block.py        # GNU Radio encoder block
│   └── decoder_block.py        # GNU Radio decoder block
│
├── flowgraphs/
│   ├── ml_comm_flowgraph.py    # Python flowgraph (no GUI)
│   └── README.md               # GRC GUI instructions
│
├── utils/
│   ├── channel_models.py       # AWGN, Rayleigh, Rician
│   └── metrics.py              # BER, SNR calculations
│
├── test_system.py              # End-to-end test script
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
```

---

## Need Help?

1. Check the README files in each directory
2. Review the code comments
3. Test individual components first
4. Start with high SNR (20 dB) and reduce gradually

**Happy experimenting! 🚀**
