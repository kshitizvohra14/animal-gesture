


import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


def main():

    parser = argparse.ArgumentParser(
        description="Train YOLO11n-Pose on Dog-Pose dataset"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size"
    )

    parser.add_argument(
        "--batch",
        type=int,
        choices=[4, 8],
        default=4,
        help="Batch size: 4 or 8"
    )

    parser.add_argument(
        "--device",
        default=None,
        help="GPU device, e.g. 0, or cpu"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers. 0 is safest on Windows."
    )

    parser.add_argument(
        "--base_model",
        default="yolo11n-pose.pt",
        help="Pretrained YOLO pose checkpoint"
    )

    args = parser.parse_args()

    # =========================================================
    # DEVICE
    # =========================================================

    if args.device is not None:
        device = args.device

    elif torch.cuda.is_available():
        device = "0"

    else:
        device = "cpu"

    print("\n" + "=" * 65)
    print("DOG POSE TRAINING")
    print("=" * 65)

    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    print(f"Training device : {device}")

    if torch.cuda.is_available():

        print(
            f"GPU             : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory      : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    print(f"Epochs          : {args.epochs}")
    print(f"Image size      : {args.imgsz}")
    print(f"Batch size      : {args.batch}")
    print(f"Workers         : {args.workers}")

    print("=" * 65)

    # =========================================================
    # CUDA CHECK
    # =========================================================

    if device != "cpu" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU was requested, but PyTorch cannot access CUDA."
        )

    # Clear unused CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # =========================================================
    # LOAD MODEL
    # =========================================================

    print(f"\nLoading {args.base_model}...")

    model = YOLO(args.base_model)

    print("Model loaded successfully.")

    # =========================================================
    # TRAIN
    # =========================================================

    print("\nStarting training...\n")

    results = model.train(

        # Dataset
        data="dog-pose.yaml",

        # Training
        epochs=args.epochs,

        # Image size
        imgsz=args.imgsz,

        # GPU
        device=device,

        # Batch
        batch=args.batch,

        # Windows
        workers=args.workers,

        # Mixed precision
        amp=True,

        # Save model
        save=True,

        # Validation
        val=True,

        # Generate plots
        plots=True,

        # Reproducibility
        seed=42,

        # Don't cache dataset into RAM/VRAM
        cache=False,

        # Automatically choose optimizer
        optimizer="auto",

        # Stop if validation stops improving
        patience=20,

        # Disable mosaic near the end
        close_mosaic=10,

        # Verbose training output
        verbose=True,
    )

    # =========================================================
    # FIND BEST MODEL
    # =========================================================

    best = Path(results.save_dir) / "weights" / "best.pt"

    if not best.exists():

        raise FileNotFoundError(
            f"Best model not found at:\n{best}"
        )

    # =========================================================
    # COPY BEST MODEL
    # =========================================================

    dest_dir = Path(__file__).parent / "models"

    dest_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dest = dest_dir / "dog_pose_best.pt"

    shutil.copy2(
        best,
        dest
    )

    # =========================================================
    # COMPLETE
    # =========================================================

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print(f"Training directory : {results.save_dir}")
    print(f"Best weights       : {best}")
    print(f"Copied to          : {dest}")

    print("\nModel ready for Paw Fussion.")

    print("\nRun your application with:")
    print("python app.py")

    print("=" * 65)


if __name__ == "__main__":
    main()