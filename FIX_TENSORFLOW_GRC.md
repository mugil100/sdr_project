# Fix: TensorFlow Not Found in GNU Radio

## Problem
GNU Radio Companion shows: `"No module named 'tensorflow'"` in Python blocks.

## Cause
GNU Radio is using a different Python interpreter than where you installed TensorFlow.

---

## Solution: Install TensorFlow in GNU Radio's Python

### Step 1: Find GNU Radio's Python

Open a terminal and run:

```bash
# Method 1: Check GRC's Python
python -c "import sys; print(sys.executable)"

# Method 2: Check what Python GRC uses
gnuradio-config-info --prefix
```

### Step 2: Install TensorFlow in That Python

**Option A: If using system Python (most common):**

```bash
pip install tensorflow>=2.13.0 numpy>=1.24.0
```

**Option B: If GNU Radio has its own Python:**

```bash
# Find the pip for GNU Radio's Python
which pip3
# or
python3 -m pip --version

# Install TensorFlow
python3 -m pip install tensorflow>=2.13.0 numpy>=1.24.0
```

**Option C: If using conda/Anaconda:**

```bash
conda install tensorflow
# or
conda install -c conda-forge tensorflow
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

You should see: `TensorFlow version: 2.20.0` (or similar)

### Step 4: Restart GNU Radio Companion

Close and reopen GRC completely, then try loading your flowgraph again.

---

## Alternative: Test Python Environment from GRC

Create a simple test block in GRC:

1. Add **Python Block**
2. In the Code section, paste:

```python
import numpy as np
from gnuradio import gr

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Python Test',
            in_sig=None,
            out_sig=None
        )
        
        # Test imports
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} found!")
        except ImportError as e:
            print(f"✗ TensorFlow not found: {e}")
            print(f"Python executable: {import sys; sys.executable}")
    
    def work(self, input_items, output_items):
        return 0
```

3. Generate the flowgraph (F5)
4. Check the console output

---

## If Installation Fails: Workaround

If you can't install TensorFlow in GRC's Python, use the **Python script method** instead:

### Option: Run Python Script Directly

```bash
cd "C:\Users\vasan\OneDrive\Desktop\SDR ASSIGNMENT"
python quick_test.py
```

This uses your system Python which already has TensorFlow!

Or run the flowgraph Python script:

```bash
cd flowgraphs
python ml_comm_flowgraph.py --snr 10 --samples 100
```

This bypasses GRC entirely but still uses GNU Radio Python libraries.

---

## Windows-Specific Solution

If on Windows, GNU Radio might be using a conda environment:

### Check conda environments:
```bash
conda env list
```

### Install in the GNU Radio environment:
```bash
conda activate gnuradio  # or whatever the GRC environment is called
pip install tensorflow>=2.13.0
```

---

## Verification Checklist

After installation, verify:

```bash
# 1. Check Python can import TensorFlow
python -c "import tensorflow; print('✓ TensorFlow OK')"

# 2. Check GNU Radio can import TensorFlow
python -c "import gnuradio; import tensorflow; print('✓ Both OK')"

# 3. Check versions match
python -c "import sys; print('Python:', sys.version)"
python -c "import tensorflow as tf; print('TF:', tf.__version__)"
```

All should work without errors!

---

## Still Having Issues?

If none of the above works, **use Method 2** from the GRC guide:

Run the Python flowgraph script directly (no GRC GUI needed):

```bash
cd flowgraphs
python ml_comm_flowgraph.py
```

This will:
- Use your system Python (which has TensorFlow)
- Still run a GNU Radio flowgraph
- Test your ML communication system
- Show results

Expected output:
```
✓ Encoder loaded successfully!
✓ Decoder ready!
Running flowgraph...
✓ Accuracy: 100.0% (100/100 correct)
```

---

## Quick Fix Summary

**Fastest solution:**
```bash
# Install TensorFlow in system Python
pip install tensorflow>=2.13.0

# Test it works
python quick_test.py

# If GRC still has issues, use Python script instead
cd flowgraphs
python ml_comm_flowgraph.py
```

This gets your system working immediately without debugging GRC's Python environment!
