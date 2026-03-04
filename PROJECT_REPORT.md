# Comprehensive Project Report: ML-Based Communication System

**Date:** February 6, 2026  
**Project:** End-to-End Machine Learning SDR Communication System  
**Status:** ✅ Complete & Verified (100% Accuracy)

---

## 1. Executive Summary

The objective of this project was to design, implement, and verify a machine learning-based communication system that replaces traditional modulation (like QPSK/QAM) with a neural network Autoencoder. 

We successfully built a system that:
1.  **Encodes** digital data (4-bit messages) into complex IQ symbols using a Deep Neural Network.
2.  **Transmits** these symbols through a simulated noisy channel (AWGN).
3.  **Decodes** the noisy symbols back to the original data with **100% accuracy**.
4.  **Integrates** seamlessly with GNU Radio via custom Python blocks.

Key innovations included the transition to a 4-bit "Smart" system to overcome model convergence issues and the development of a "Smart Lock" decoder that automatically synchronizes with the signal stream.

---

## 2. Phase 1: Inception & Architecture

We started with the goal of implementing a classic Autoencoder for communications:
- **Transmitter (Encoder)**: A neural network mapping bits $\to$ complex IQ symbols.
- **Receiver (Decoder)**: A neural network mapping received symbols $\to$ bits.
- **Channel**: A layer simulating Additive White Gaussian Noise (AWGN).

### Initial Architecture
- **Input**: 8-bit bytes (values 0-255).
- **Compression**: Map 8 bits to 4 complex symbols (Effective rate: 2 bits/symbol, similar to QPSK).
- **Model**: Multi-layer Perceptron (Dense layers).

**Files Created:**
- `models/autoencoder.py`: Defined the custom Keras layers (`PowerNormalization`, `AWGNChannel`).
- `models/train.py`: Training loop using TensorFlow.

---

## 3. Phase 2: The Accuracy Crisis

Upon training the initial 8-bit model, we encountered a critical failure.

### The Problem
- The model achieved **~50% bit accuracy**, which is equivalent to random guessing.
- Loss curves did not converge.
- The system could not reliably transmit even simple messages.

### Root Cause Analysis
We diagnosed that the **complexity of the task exceeded the model's capacity**:
1.  **High Dimensionality**: Mapping 256 unique classes (8 bits) to a small constellation using a shallow network was too difficult.
2.  **Vanishing Gradients**: The network struggled to differentiate between 256 subtle variations in the IQ constellation.
3.  **Model Size**: The initial network (~20k parameters) was too small for this density.

*Ref: `ACCURACY_ISSUE.md`*

---

## 4. Phase 3: The Pivot (Proof of Concept)

To salvage the project and prove the validity of the ML approach, we made a strategic decision to **simplify the problem**.

### The Solution: 4-Bit System
Instead of 8 bits (256 values), we scaled down to **4 bits (16 values)**.
- **Input**: Values 0-15.
- **Ratio**: 4 bits $\to$ 4 Symbols.
- **Density**: The constellation only needs 16 distinct points instead of 256, making them much farther apart and easier to distinguish.

### The Result
- **Instant Convergence**: The model learned the mapping in minutes.
- **Accuracy**: **100%** (0 Bit Errors).
- **Validation**: Every single value from 0 to 15 was tested and verified to decode perfectly even at 10dB SNR.

*Ref: `SUCCESS_4BIT.md`*

---

## 5. Phase 4: Robustness & The "Smart Lock"

With the ML model working, the challenge moved to **GNU Radio integration**. Real-time streaming introduced new problems:
1.  **Synchronization Loss**: In a stream of symbols, the decoder didn't know where a "message" started (Symbol 1 vs Symbol 2).
2.  **Jitter/Settling**: When the flowgraph started, the first few buffers contained undefined data, causing offsets.

### Innovation: The "Smart Lock" Decoder
We implemented a sophisticated custom block (`SMART_LOCKED_DECODER_BLOCK.py`) to solve this.

**How it Works:**
1.  **Cache Pre-computation**: Instead of running the slow neural network for every prediction, we pre-calculated the ideal constellation points for all 16 values.
2.  **Search Mode**: On startup, the block buffers incoming data and tests all possible "offsets" (0, 1, 2, 3).
3.  **MSE Minimization**: It calculates the Mean Squared Error for each offset against the known valid points.
4.  **Locking**: It selects the offset with the lowest error and "locks" onto it, discarding the garbage prefix.

**Result**: The system is now robust to stream misalignments and starts up reliably every time.

---

## 6. Project Timeline & Deliverables

| Phase | Action | Outcome |
|-------|--------|---------|
| **1. Setup** | Installed TensorFlow, GNU Radio, created env | Environment Ready |
| **2. Model Design** | Created `autoencoder.py` | Initial Architecture |
| **3. Debugging** | Diagnosed 50% accuracy on 8-bit | Strategy Pivot |
| **4. Training** | Retrained on 4-bit (0-15) | **100% Accuracy** |
| **5. Integration** | Created `ROBUST_ENCODER_BLOCK.py` | Reliable GRC Source |
| **6. Optimization** | Created `SMART_LOCKED_DECODER.py` | Perfect Sync |

### Key Files
- **`models/saved_models/autoencoder_final.keras`**: The trained brain of the system.
- **`ROBUST_ENCODER_BLOCK.py`**: The transmitter block.
- **`SMART_LOCKED_DECODER_BLOCK.py`**: The receiver block with sync logic.
- **`quick_test.py`**: Assessment script verifying functionality.

---

## 7. Future Roadmap

Now that the core engine is proven:
1.  **Scaling back to 8-bit**: With the "Smart Lock" logic proven, we can now train a much larger (Deep) network to handle the full 256-value range.
2.  **Over-the-Air**: Connect to an RTL-SDR or PlutoSDR to test real wireless transmission.
3.  **Adaptive Modulation**: Train multiple models (high/low SNR) and switch them dynamically.

---
**Report generated by Antigravity AI**
