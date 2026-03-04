# Installing TensorFlow in Radioconda (GNU Radio)

## ✅ Great! You have Radioconda installed!

Radioconda is a conda-based distribution that includes GNU Radio with its own Python environment.

---

## Quick Fix - Install TensorFlow in Radioconda

### Step 1: Open Radioconda Prompt

**Option A: From Start Menu**
1. Click Start
2. Search for "**Radioconda Prompt**" or "**Anaconda Prompt (radioconda)**"
3. Click to open it

**Option B: From PowerShell**
```powershell
# Initialize conda
conda init powershell

# Close and reopen PowerShell, then:
conda activate radioconda
```

### Step 2: Install TensorFlow

In the Radioconda Prompt (or activated conda environment), run:

```bash
# Install TensorFlow
pip install tensorflow>=2.13.0

# Or use conda (may be slower):
conda install -c conda-forge tensorflow
```

**This should take 2-5 minutes.**

### Step 3: Verify Installation

Still in Radioconda Prompt:

```bash
python -c "import tensorflow as tf; print('✓ TensorFlow', tf.__version__, 'installed!')"
python -c "import gnuradio; import tensorflow; print('✓ Both work together!')"
```

You should see both commands succeed!

### Step 4: Restart GNU Radio Companion

1. **Close** any open GRC windows
2. From Radioconda Prompt, launch GRC:
   ```bash
   gnuradio-companion
   ```

3. **Open your flow graph** and try generating it again

The TensorFlow error should be gone!

---

## Detailed Steps with Screenshots

### Opening Radioconda Prompt

1. **Press Windows Key**
2. **Type**: `radioconda`
3. You should see: **"Anaconda Prompt (radioconda)"** or **"Radioconda Prompt"**
4. **Click it**

You'll see a prompt like:
```
(radioconda) C:\Users\vasan>
```

The `(radioconda)` prefix means you're in the right environment!

### Install Command

In that prompt, simply run:

```bash
pip install tensorflow
```

Wait for it to complete. You'll see:
```
Collecting tensorflow
  Downloading tensorflow-2.x.x...
Installing collected packages: ...
Successfully installed tensorflow-2.x.x
```

### Test It Works

```bash
python -c "import tensorflow; print('✓ Works!')"
```

Success!

---

## Alternative: If You Can't Find Radioconda Prompt

### Find Radioconda Installation

Check if these folders exist:
- `C:\radioconda`
- `C:\ProgramData\radioconda`  
- `C:\Users\vasan\radioconda`

Once you find it, navigate there in PowerShell:

```powershell
cd C:\radioconda  # or wherever it is

# Activate the environment
.\Scripts\activate

# Install TensorFlow
pip install tensorflow
```

---

## Quick Command Summary

**Fastest method (copy-paste into Radioconda Prompt):**

```bash
pip install tensorflow>=2.13.0 && python -c "import tensorflow as tf; print('✓ TensorFlow', tf.__version__)" && echo "Now restart GNU Radio Companion!"
```

This:
1. Installs TensorFlow
2. Verifies it works
3. Reminds you to restart GRC

---

## After Installation

### Restart GRC:
```bash
# In Radioconda Prompt:
gnuradio-companion
```

### Try your flowgraph again!

The Python block should now compile without the TensorFlow error.

---

## Still Having Issues?

If you see an error when running `conda env list` or `pip install tensorflow`, share the error message and I'll help you fix it!

Common issues:
- **"conda not found"**: Need to initialize conda first
- **"Permission denied"**: Run Radioconda Prompt as Administrator
- **"Package conflicts"**: Use `pip install --no-deps tensorflow` instead
