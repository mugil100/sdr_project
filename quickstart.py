#!/usr/bin/env python3
"""
Quick Start Script for ML Communication System
Automates the entire setup and testing process
"""

import os
import sys
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def print_step(step_num, total_steps, description):
    """Print a step indicator"""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 70)


def run_command(cmd, cwd=None, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"Running: {description}")
    print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=True,
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ Command failed with return code {result.returncode}")
            return False
        
        print(f"\n✓ {description if description else 'Command'} completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def check_file_exists(filepath, description=""):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description if description else filepath}")
        return True
    else:
        print(f"✗ {description if description else filepath} - NOT FOUND")
        return False


def main():
    print_header("ML-Based Communication System - Quick Start")
    print("This script will:")
    print("  1. Install Python dependencies")
    print("  2. Train the autoencoder model")
    print("  3. Run end-to-end tests")
    print("  4. Display results")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    total_steps = 5
    
    # Step 1: Install dependencies
    print_step(1, total_steps, "Installing Python dependencies")
    
    if not run_command(
        f"pip install -r requirements.txt",
        description="Installing TensorFlow, NumPy, Matplotlib, SciPy"
    ):
        print("\n⚠ Installation may have had issues. Continuing anyway...")
    
    # Step 2: Verify installation
    print_step(2, total_steps, "Verifying installation")
    
    try:
        import numpy
        import tensorflow
        import matplotlib
        import scipy
        print("✓ NumPy version:", numpy.__version__)
        print("✓ TensorFlow version:", tensorflow.__version__)
        print("✓ Matplotlib version:", matplotlib.__version__)
        print("✓ SciPy version:", scipy.__version__)
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return 1
    
    # Try to import gnuradio (optional for training)
    try:
        import gnuradio
        print("✓ GNU Radio version:", gnuradio.version.version())
    except ImportError:
        print("⚠ GNU Radio not found in Python path")
        print("  (This is OK for training, but needed for flowgraph execution)")
    
    # Step 3: Train the model
    print_step(3, total_steps, "Training the autoencoder model")
    print("This may take 5-15 minutes depending on your hardware...")
    print("Training with:")
    print("  - 100,000 training samples")
    print("  - 8 bits per sample (0-255 range)")
    print("  - 4 complex symbols output")
    print("  - 10 dB SNR")
    
    models_dir = os.path.join(script_dir, "models")
    
    if not run_command(
        "python train.py",
        cwd=models_dir,
        description="Training autoencoder"
    ):
        print("\n❌ Training failed!")
        print("Please check the error messages above.")
        return 1
    
    # Verify trained models exist
    print("\nVerifying trained models...")
    saved_models_dir = os.path.join(models_dir, "saved_models")
    
    models_ok = True
    models_ok &= check_file_exists(
        os.path.join(saved_models_dir, "encoder.keras"),
        "Encoder model"
    )
    models_ok &= check_file_exists(
        os.path.join(saved_models_dir, "decoder.keras"),
        "Decoder model"
    )
    models_ok &= check_file_exists(
        os.path.join(saved_models_dir, "autoencoder_final.keras"),
        "Full autoencoder model"
    )
    models_ok &= check_file_exists(
        os.path.join(saved_models_dir, "training_history.png"),
        "Training history plot"
    )
    
    if not models_ok:
        print("\n❌ Some model files are missing!")
        return 1
    
    # Step 4: Run tests
    print_step(4, total_steps, "Running end-to-end tests")
    print("Testing the complete communication pipeline...")
    
    if not run_command(
        "python test_system.py --snr 10",
        description="Testing encoder → channel → decoder"
    ):
        print("\n❌ Test failed!")
        return 1
    
    # Step 5: Summary
    print_step(5, total_steps, "Setup Complete!")
    
    print("\n🎉 SUCCESS! Your ML communication system is ready!")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    print("\n1. View training results:")
    print(f"   {os.path.join(saved_models_dir, 'training_history.png')}")
    print(f"   {os.path.join(saved_models_dir, 'ber_vs_snr.png')}")
    
    print("\n2. Test at different SNR levels:")
    print("   python test_system.py --snr 5    # Noisy")
    print("   python test_system.py --snr 15   # Clean")
    
    print("\n3. Run the GNU Radio flowgraph:")
    print("   cd flowgraphs")
    print("   python ml_comm_flowgraph.py")
    
    print("\n4. Create a visual flowgraph in GRC:")
    print("   gnuradio-companion")
    print("   See flowgraphs/README.md for instructions")
    
    print("\n5. Read the documentation:")
    print("   - README.md - Project overview")
    print("   - GETTING_STARTED.md - Detailed guide")
    print("   - flowgraphs/README.md - GRC integration")
    
    print("\n" + "=" * 70)
    print("For detailed help, see GETTING_STARTED.md")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
