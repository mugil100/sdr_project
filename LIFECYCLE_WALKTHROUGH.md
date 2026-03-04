# Lifecycle of a Number: Step-by-Step Walkthrough

**Date:** February 6, 2026  
**Example Value:** `14` (Decimal)  
**System:** 4-Bit Machine Learning Transceiver

---

## ❓ Critical Concept: The Mapping Logic

**"Is it 14 $\to$ 4 symbols? Or each bit $\to$ 4 symbols?"**

It is **14 $\to$ 4 symbols**.

In this system, we use **Block Coding** (Dense Layers):
*   **Input:** The *entire group* of 4 bits (`1110`).
*   **Output:** The *entire group* of 4 symbols.

Every bit affects every symbol. If you change even one bit (e.g., `1110` $\to$ `0110`), **ALL four symbols** will likely change completely to a new set of values. This "scrambling" makes the signal much more robust against noise.

---

## 1. The Journey Begins (Input)

We start with the user wanting to send the number **14**.

Since our Learning Model works with digital signals, we first convert this to binary. In our 4-bit system, 14 is represented as:

$$ 14_{10} = 1110_{2} $$

**Data State:** `[1, 1, 1, 0]`

---

## 2. The Neural Transmitter (Encoder)

This binary vector is fed into the **Encoder** neural network. The network has learned that to protect this specific pattern (`1110`) from noise, it should be mapped to a specific set of 4 complex coordinates.

**Action:** The Neural Network multiplies the input by its weights (matrices) and applies non-linear activation functions.

**Output (Real-Time Captured Values):**
The encoder outputs 4 complex **IQ Symbols**. These are the coordinates in the signal space.

| Symbol | Real Part (I) | Imaginary Part (Q) | Notation |
| :--- | :--- | :--- | :--- |
| **0** | `-1.1747` | `+0.2055j` | $S_0$ |
| **1** | `-0.2768` | `+0.7121j` | $S_1$ |
| **2** | `-1.0870` | `+0.5374j` | $S_2$ |
| **3** | `-0.4701` | `-0.5503j` | $S_3$ |

*Observation: Notice the values are not just +1/-1 like digital logic. They are precise analog voltages.*

---

## 3. The Dangerous Road (The Channel)

The symbols travel through the "air" (or wire). In reality, air is noisy. We simulate this by adding random **Gaussian Noise** to each symbol.

$$ \text{Received} = \text{Transmitted} + \text{Noise} $$

**Noise Added (Simulation):**
- Symbol 0 got hit by `0.05 - 0.05j`
- Symbol 1 got hit by `-0.02 + 0.08j`
- ...and so on.

**The Corrupted Signal (What the Receiver sees):**

| Symbol | Transmitted | **Received (Noisy)** | Drift |
| :--- | :--- | :--- | :--- |
| **0** | `-1.17` + `0.20j` | **`-1.12` + `0.15j`** | ↘️ |
| **1** | `-0.27` + `0.71j` | **`-0.29` + `0.79j`** | ↖️ |
| **2** | `-1.08` + `0.53j` | **`-0.98` + `0.54j`** | ➡️ |
| **3** | `-0.47` - `0.55j` | **`-0.51` - `0.59j`** | ↙️ |

*The points have moved! They are no longer in their perfect locations.*

---

## 4. The Neural Receiver (Decoder)

The **Decoder** neural network receives these messy, shifted points. It doesn't know the original value was 14. It only sees the "Received" column above.

**Action:**
1.  It plots the received points in 4D space.
2.  It asks: *"Which of the 16 patterns is this closest to?"*
3.  It calculates the probability for each possible bit.

**Decoder Confidence (Softmax/Sigmoid):**
*   **Bit 0:** `99.5%` confident it is a **1**
*   **Bit 1:** `91.5%` confident it is a **1**
*   **Bit 2:** `95.0%` confident it is a **1**
*   **Bit 3:** `12.0%` confident it is a **1** (So it's a **0**)

**Reconstruction:** `[1, 1, 1, 0]`

---

## 5. The Destination (Output)

The system converts the reconstructed binary back to decimal.

$$ 1110_{2} \rightarrow 14_{10} $$

**Result:**
*   **Sent:** 14
*   **Received:** 14
*   **Status:** ✅ Perfect Transmission

### Summary
The magic is in **Step 2 and Step 4**. The Neural Networks effectively agreed on a "secret language" (the constellation coordinates) that allowed the message `14` to survive the distortion in **Step 3**.
