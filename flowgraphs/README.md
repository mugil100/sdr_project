# GNU Radio Flowgraph Guide

This directory will contain the GNU Radio Companion (.grc) flowgraphs for the ML-based communication system.

## Quick Start

Since GNU Radio Companion uses a graphical interface, you'll need to create the flowgraph manually. Here's how:

### Step 1: Creating the Basic Flowgraph

1. Open GNU Radio Companion (run `gnuradio-companion` in terminal)

2. **Add these blocks** (search for them in the block browser):

   - **Options** (already present)
     - Set Title: "ML Communication System"
     - Set Author: Your name
     
   - **Variable** blocks:
     - `samp_rate`: 32000
     - `snr_db`: 10 (you can adjust this)
   
   - **Vector Source** (for generating test data):
     - Output Type: Byte
     - Vector: `[23, 45, 100, 200, 255, 0, 128, 42]` (or any values 0-255)
     - Repeat: Yes
   
   - **Throttle**:
     - Type: Byte
     - Sample Rate: `samp_rate`
   
   - **Python Module** or **Embedded Python Block**:
     - This is where you'll integrate the ML encoder
   
   - **Channel Model** (from Channels category):
     - Noise Voltage: Calculate from SNR
     - Seed: 0
   
   - **Another Python Module/Block** for decoder
   
   - **QT GUI Constellation Sink** (to visualize symbols):
     - Type: Complex
     - Number of Points: 1024
   
   - **QT GUI Time Sink** (to see signals over time):
     - Type: Complex
   
   - **Vector Sink** or **File Sink** (to save output)

### Step 2: Using Custom Python Blocks

Unfortunately, GNU Radio 3.10 doesn't easily support importing custom blocks from external files in the GUI. 

**You have two options:**

#### Option A: Use Embedded Python Block (Simpler)

1. Add "Embedded Python Block" from the block browser

2. **For Encoder Block:**
   ```python
   import numpy as np
   from gnuradio import gr
   import tensorflow as tf
   import os
   
   class ml_encoder_embedded(gr.sync_block):
       def __init__(self):
           gr.sync_block.__init__(
               self,
               name="ML Encoder",
               in_sig=[np.uint8],
               out_sig=[np.complex64]
           )
           model_path = "C:/Users/vasan/OneDrive/Desktop/SDR ASSIGNMENT/models/saved_models/encoder.keras"
           self.encoder = tf.keras.models.load_model(model_path, compile=False)
           self.set_output_multiple(4)
       
       def work(self, input_items, output_items):
           in_bytes = input_items[0]
           out_symbols = output_items[0]
           
           num_to_process = min(len(in_bytes), len(out_symbols) // 4)
           if num_to_process == 0:
               return 0
           
           bits_batch = []
           for i in range(num_to_process):
               byte_val = int(in_bytes[i])
               bits = [int(b) for b in format(byte_val, '08b')]
               bits_batch.append(bits)
           
           bits_array = np.array(bits_batch, dtype=np.float32)
           encoded_symbols = self.encoder.predict(bits_array, verbose=0)
           out_symbols[:num_to_process * 4] = encoded_symbols.flatten()
           
           return num_to_process * 4
   ```

#### Option B: Use Python Snippet Block (Advanced)

Create a hierarchical block or use the Python Snippet block to import your custom modules.

### Step 3: Simplified Testing Flowgraph

For initial testing, create this simple flowgraph:

```
[Vector Source] → [Throttle] → [File Sink: input_bytes.dat]
                              ↓
                     [Custom Encoder Block]
                              ↓
                     [QT Constellation Sink]
                              ↓
                     [Channel Model (AWGN)]
                              ↓
                     [QT Constellation Sink]
                              ↓
                     [Custom Decoder Block]
                              ↓
                     [File Sink: output_bytes.dat]
```

### Step 4: Alternative - Pure Python Flowgraph

Create a pure Python flowgraph (recommended for ML blocks):

See `ml_comm_flowgraph.py` in this directory for a complete Python-based flowgraph that doesn't require the GUI.

## Running the Flowgraph

### Method 1: Python Script (Recommended)
```bash
cd flowgraphs
python ml_comm_flowgraph.py
```

### Method 2: GNU Radio Companion
```bash
gnuradio-companion
# File → Open → select your .grc file
# Click "Execute" button (▶)
```

## Troubleshooting

**Issue**: "Module 'tensorflow' not found"
- **Solution**: Make sure you've installed dependencies: `pip install -r requirements.txt`

**Issue**: "Model file not found"
- **Solution**: Train the model first: `cd models && python train.py`

**Issue**: "gnuradio module not found"
- **Solution**: Ensure GNU Radio is properly installed and Python can find it

**Issue**: Custom blocks not showing in GRC
- **Solution**: Use Embedded Python Blocks instead, or create a proper OOT module

## Next Steps

1. First, test the system using the Python test script: `python test_system.py`
2. Then try the pure Python flowgraph: `python flowgraphs/ml_comm_flowgraph.py`
3. Finally, create a GRC flowgraph following the instructions above

For more advanced usage, consider creating an Out-of-Tree (OOT) module for proper GRC integration.
