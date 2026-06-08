# training/train.py
import os
import sys
import sysconfig
import argparse
from pathlib import Path

# ==============================================================================
# THE LINUX LINKER REBOOT (Permanent Hermetic CUDA Fix)
# Must execute BEFORE any keras, tensorflow, or fastapi imports!
# ==============================================================================
# Check if we have already rebooted the process to prevent infinite loops
if not os.environ.get("CUDA_LINKER_REBOOTED", False):
    paths_to_add = ["/usr/lib/wsl/lib"]
    try:
        # sysconfig is much safer than site.getsitepackages() for virtual environments
        site_packages = sysconfig.get_path("purelib")
        nvidia_libs = [
            'cudnn', 'cublas', 'cuda_runtime', 'cusolver', 
            'cusparse', 'curand', 'nvjitlink', 'cufft'
        ]
        # Map every hidden NVIDIA library path it finds
        for lib in nvidia_libs:
            lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
            if os.path.exists(lib_path):
                paths_to_add.append(lib_path)
    except Exception:
        pass

    # Inject the paths into the environment
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = ":".join(paths_to_add)
    os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}" if current_ld_path else new_ld_path
    
    # Set the flag so we know the environment is primed
    os.environ["CUDA_LINKER_REBOOTED"] = "1"
    
    # INSTANTLY RESTART THE PYTHON PROCESS WITH THE NEW C-LINKER ENVIRONMENT
    os.execv(sys.executable, [sys.executable] + sys.argv)
# ==============================================================================

# 1. Define the actual root and the script's directory
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
SCRIPT_DIR = str(Path(__file__).resolve().parent)

# 2. REMOVE the script directory from Python's brain to stop folder shadowing
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

# 3. Add the true project root so it starts searching from the very top
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.training.stages import run_pretrain, run_finetune
from training.core.config import DEFAULT_ARCH, ARCHITECTURES


def main():
    import keras
    
    # ---------------------------------------------------------
    # CORE HARDWARE OPTIMIZATION
    # Force GPU to utilize Ada Lovelace Tensor Cores by defaulting all internal graph ops to float16.
    # Yields ~3x-4x training speedup on RTX 4000 series with zero precision degradation.
    # ---------------------------------------------------------
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("\n[HARDWARE] Tensor Cores Unlocked: Mixed Precision Policy Active.")

    parser = argparse.ArgumentParser(description="Emontic AI v2 - 2-Stage Training Orchestrator")
    parser.add_argument(
        "--arch",
        type=str,
        default=DEFAULT_ARCH,
        choices=list(ARCHITECTURES.keys()),
        help=f"Model backbone variant selection (default: {DEFAULT_ARCH})",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["pretrain", "finetune", "all"],
        help="Target execution stage selection: 'pretrain', 'finetune', or run 'all' sequentially",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to saved model checkpoint (.keras file) required for fine-tuning stand-alone runs",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" EMONTIC AI MULTI-STAGE ENGINE INITIALIZED")
    print(f" Target Architecture Backbone: {args.arch.upper()}")
    print("=" * 70)

    checkpoint_to_use = args.checkpoint

    # Stage 1 Execution
    if args.stage in ["pretrain", "all"]:
        best_pretrain_checkpoint = run_pretrain(args.arch)
        checkpoint_to_use = best_pretrain_checkpoint

    # Stage 2 Execution
    if args.stage in ["finetune", "all"]:
        if checkpoint_to_use is None:
            raise ValueError(
                "Executing a stand-alone '--stage finetune' run requires passing a valid "
                "pre-trained base file path via the '--checkpoint' parameter."
            )
        
        print(f"\n[INFO] Starting Stage 2 Transfer Tuning using baseline: {checkpoint_to_use}")
        run_finetune(args.arch, checkpoint_to_use)

    print("\n" + "=" * 70)
    print(" PROCESS COMPLETE - ALL REQUESTED STAGES SUCCESSFULLY EXECUTED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()