# Step-by-Step: Fix TensorFlow in GNU Radio Companion

## ✅ Good News!
TensorFlow 2.20.0 is already installed in your Python 3.10!

## The Issue
GNU Radio Companion sometimes caches Python module information. Even though TensorFlow is installed, GRC might not see it until you:

1. **Restart GRC completely**
2. **Clear GRC's cache**
3. **Regenerate all blocks**

---

## Fix Steps

### Step 1: Close GNU Radio Companion
- Close GRC completely (File → Quit)
- Make sure no GRC windows are open

### Step 2: Clear GRC Cache (Optional but Recommended)

Delete the GRC cache directory:

```powershell
# Navigate to your user directory
cd ~

# Remove GRC cache (it will be recreated)
Remove-Item -Recurse -Force .gnuradio
```

Or manually delete: `C:\Users\vasan\.gnuradio\`

### Step 3: Reopen GNU Radio Companion

```powershell
cd "C:\Users\vasan\OneDrive\Desktop\SDR ASSIGNMENT"
gnuradio-companion
```

### Step 4: Open Your Flowgraph

1. File → Open
2. Select your flowgraph file
3. Click **Generate** (F5)

**It should work now!**

---

## If It Still Shows the Error...

### Option A: Simplify the Import

In your Python block code, try importing TensorFlow differently:

**Instead of:**
```python
import tensorflow as tf
```

**Try:**
```python
try:
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} loaded!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    import sys
    print(f"Python path: {sys.path}")
    raise
```

This will give you more detailed error information.

### Option B: Add Python Path Explicitly

At the very top of your Python block code, add:

```python
import sys
sys.path.insert(0, r'C:\Users\vasan\AppData\Local\Programs\Python\Python310\Lib\site-packages')

import numpy as np
from gnuradio import gr
import tensorflow as tf
```

This explicitly tells Python where to find TensorFlow.

### Option C: Check GRC's Python

Create a simple test block to print Python info:

```python
import numpy as np
from gnuradio import gr
import sys

class blk(gr.sync_block):
    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Test Block',
            in_sig=None,
            out_sig=None
        )
        
        print("=" * 60)
        print("Python Environment Info:")
        print(f"Python executable: {sys.executable}")
        print(f"Python version: {sys.version}")
        print(f"Python path: {sys.path[:3]}")
        
        try:
            import tensorflow as tf
            print(f"✓ TensorFlow {tf.__version__} found!")
        except ImportError as e:
            print(f"✗ TensorFlow NOT found: {e}")
        
        print("=" * 60)
    
    def work(self, input_items, output_items):
        return 0
```

Generate and run this - check what it prints in the console.

---

## Quick Test

Before opening GRC, verify everything works:

```powershell
# Test in PowerShell
python -c "import gnuradio; import tensorflow as tf; print('✓ Both work! TensorFlow:', tf.__version__)"
```

If this works, GRC should work too after restarting.

---

## Most Likely Solution

**99% of the time, this fixes it:**

1. Close GRC completely
2. Wait 5 seconds  
3. Reopen GRC
4. Try generating your flowgraph again

GRC caches module information, and restarting clears that cache!

---

## Need More Help?

If you're still stuck:

1. Take a screenshot of the exact error in GRC
2. Run this command and share the output:
   ```powershell
   python -c "import sys; print('\n'.join(sys.path))"
   ```
3. Check if there are multiple Python installations:
   ```powershell
   where python
   ```

Then I can provide more specific help!
