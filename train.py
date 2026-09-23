
"""
train.py
--------
MobileNetV2 pet-emotion classifier.

Dataset structure expected:

    data/
    ├── train/
    │   ├── Class1/
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   ├── Class2/
    │   └── ...
    │
    └── val/
        ├── Class1/
        ├── Class2/
        └── ...

Features:
    - Uses existing train/val folders directly
    - Class names are read from train/ and verified against val/
    - Corrupted-image pre-flight scan
    - Optional quarantine of corrupted images
    - Class-weight balancing
    - MobileNetV2 transfer learning
    - Two-phase training when starting from scratch
    - Resume/fine-tune an existing .keras model
    - Early stopping
    - ReduceLROnPlateau
    - Best-model checkpointing
    - Per-class precision/recall/F1
    - Confusion matrix

Examples:

Start a new model:

    python train.py --dataset data

Resume an existing model:

    python train.py --dataset data --resume pet_emotion_mobilenetv2.keras

Resume and fine-tune more aggressively:

    python train.py --dataset data ^
        --resume pet_emotion_mobilenetv2.keras ^
        --finetune-epochs 30 ^
        --unfreeze-layers 40 ^
        --finetune-lr 5e-6

Scan only:

    python train.py --dataset data --scan-only

Quarantine corrupted images:

    python train.py --dataset data --quarantine
"""

import argparse
import json
import os
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "data"

IMG_SIZE = 224
BATCH_SIZE = 16

HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 25

HEAD_LR = 1e-3
FINETUNE_LR = 1e-5

# Number of MobileNetV2 layers from the top to unfreeze.
UNFREEZE_LAYERS = 40

LABEL_SMOOTHING = 0.1

SEED = 42

MODEL_PATH = "pet_emotion_mobilenetv2.keras"
HEAD_MODEL_PATH = "head_only_pet_emotion_mobilenetv2.keras"

CLASS_NAMES_PATH = "class_names.json"
CLASS_INDICES_PATH = "models/class_indices.json"

CORRUPT_REPORT_PATH = "corrupted_files.txt"
QUARANTINE_DIR = "_corrupted"

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

MIN_DIMENSION = 32


# ============================================================
# DATASET DISCOVERY
# ============================================================

def find_class_names(train_dir, val_dir):
    """
    Discover classes from train/ and make sure val/ contains
    exactly the same classes.
    """

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training directory not found:\n{train_dir}"
        )

    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Validation directory not found:\n{val_dir}"
        )

    train_classes = sorted(
        entry
        for entry in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, entry))
        and not entry.startswith((".", "_"))
    )

    val_classes = sorted(
        entry
        for entry in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, entry))
        and not entry.startswith((".", "_"))
    )

    if not train_classes:
        raise RuntimeError(
            f"No class folders found inside:\n{train_dir}"
        )

    if train_classes != val_classes:
        missing_from_val = sorted(set(train_classes) - set(val_classes))
        extra_in_val = sorted(set(val_classes) - set(train_classes))

        message = "\nClass mismatch between train and val.\n"

        if missing_from_val:
            message += (
                f"\nMissing from val:\n"
                f"  {missing_from_val}\n"
            )

        if extra_in_val:
            message += (
                f"\nExtra classes in val:\n"
                f"  {extra_in_val}\n"
            )

        raise RuntimeError(message)

    return train_classes


def find_image_files(folder, class_names):
    """
    Return:
        [(path, label_index), ...]

    Label is determined by the class folder directly under
    train/ or val/.
    """

    class_to_index = {
        name: i
        for i, name in enumerate(class_names)
    }

    items = []
    skipped_extension = 0

    for class_name in class_names:

        class_dir = os.path.join(folder, class_name)

        label = class_to_index[class_name]

        for root, _, files in os.walk(class_dir):

            for filename in files:

                if filename.lower().endswith(VALID_EXTENSIONS):

                    items.append(
                        (
                            os.path.join(root, filename),
                            label,
                        )
                    )

                else:
                    skipped_extension += 1

    return items, skipped_extension


# ============================================================
# IMAGE CORRUPTION SCAN
# ============================================================

def inspect_file(path):
    """
    Fully decode an image using Pillow.
    """

    try:

        if os.path.getsize(path) == 0:
            return False, "empty file (0 bytes)"

        with Image.open(path) as probe:
            probe.verify()

        with Image.open(path) as img:

            img.load()

            width, height = img.size

            img.convert("RGB")

        if width < MIN_DIMENSION or height < MIN_DIMENSION:

            return (
                False,
                f"too small ({width}x{height}, "
                f"minimum {MIN_DIMENSION})",
            )

        return True, ""

    except Exception as exc:

        return (
            False,
            f"{type(exc).__name__}: {exc}",
        )


def scan_dataset(items, workers=8):

    paths = [
        path
        for path, _ in items
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:

        results = list(
            pool.map(inspect_file, paths)
        )

    good = []
    bad = []

    for item, result in zip(items, results):

        path, label = item
        ok, reason = result

        if ok:
            good.append(item)
        else:
            bad.append(
                (path, reason)
            )

    return good, bad


def write_corrupt_report(
    bad_records,
    dataset_dir,
    report_path,
):

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            f"# {len(bad_records)} "
            f"unreadable file(s) under "
            f"{dataset_dir}\n"
        )

        for path, reason in bad_records:

            handle.write(
                f"{path}\t{reason}\n"
            )


def quarantine(
    bad_records,
    dataset_dir,
    quarantine_dir,
):

    moved = 0

    for path, _ in bad_records:

        relative = os.path.relpath(
            path,
            dataset_dir,
        )

        destination = os.path.join(
            quarantine_dir,
            relative,
        )

        os.makedirs(
            os.path.dirname(destination),
            exist_ok=True,
        )

        try:

            shutil.move(
                path,
                destination,
            )

            moved += 1

        except OSError as exc:

            print(
                f"Could not move {path}: {exc}"
            )

    return moved


def report_scan(
    good,
    bad,
    class_names,
    skipped_extension,
    dataset_name,
):

    total = len(good) + len(bad)

    print("\n" + "=" * 60)
    print(f"PRE-FLIGHT SCAN: {dataset_name}")
    print("=" * 60)

    print(f"  files checked : {total}")
    print(f"  usable        : {len(good)}")
    print(f"  unreadable    : {len(bad)}")

    if skipped_extension:

        print(
            f"  ignored files : {skipped_extension}"
        )

    counts = {
        name: 0
        for name in class_names
    }

    for _, label in good:

        counts[
            class_names[label]
        ] += 1

    print("\n  Usable images per class:")

    for name in class_names:

        print(
            f"    {name:30s}"
            f"{counts[name]:6d}"
        )

    return counts


# ============================================================
# CLASS WEIGHTS
# ============================================================

def make_class_weights(
    items,
    num_classes,
):

    counts = [
        0
    ] * num_classes

    for _, label in items:

        counts[label] += 1

    total = sum(counts)

    return {
        i:
        total / (num_classes * count)
        for i, count in enumerate(counts)
        if count > 0
    }


# ============================================================
# TF.DATA
# ============================================================

def apply_ignore_errors(tf, dataset):

    if hasattr(dataset, "ignore_errors"):

        try:

            return dataset.ignore_errors(
                log_warning=True
            )

        except TypeError:

            return dataset.ignore_errors()

    return dataset.apply(
        tf.data.experimental.ignore_errors()
    )


def build_dataset(
    tf,
    items,
    num_classes,
    batch_size,
    shuffle,
    seed,
    img_size,
):

    paths = [
        path
        for path, _ in items
    ]

    labels = [
        label
        for _, label in items
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels,
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=min(
                len(items),
                2000,
            ),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load(path, label):

        image_bytes = tf.io.read_file(path)

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        image.set_shape(
            [None, None, 3]
        )

        image = tf.image.resize(
            image,
            [img_size, img_size],
        )

        image = tf.cast(
            image,
            tf.float32,
        )

        return (
            image,
            tf.one_hot(
                label,
                depth=num_classes,
            ),
        )

    dataset = dataset.map(
        load,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = apply_ignore_errors(
        tf,
        dataset,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    return dataset.prefetch(
        tf.data.AUTOTUNE
    )


def count_elements(dataset):

    total = 0

    for images, _ in dataset:

        total += int(
            images.shape[0]
        )

    return total


# ============================================================
# MODEL
# ============================================================

def compile_model(
    keras,
    model,
    learning_rate,
):

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=[
            "accuracy"
        ],
    )


def build_model(
    tf,
    keras,
    layers,
    num_classes,
    img_size,
    learning_rate,
):

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal"
            ),
            layers.RandomRotation(
                0.08
            ),
            layers.RandomZoom(
                0.10
            ),
            layers.RandomContrast(
                0.10
            ),
        ],
        name="data_augmentation",
    )

    base_model = (
        tf.keras.applications.MobileNetV2(
            input_shape=(
                img_size,
                img_size,
                3,
            ),
            include_top=False,
            weights="imagenet",
        )
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(
            img_size,
            img_size,
            3,
        )
    )

    x = data_augmentation(inputs)

    x = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(x)
    )

    x = base_model(
        x,
        training=False,
    )

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(
        0.30
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = keras.Model(
        inputs,
        outputs,
    )

    compile_model(
        keras,
        model,
        learning_rate,
    )

    return model, base_model


def unfreeze_top_layers(
    layers,
    base_model,
    n_layers,
):

    base_model.trainable = True

    cutoff = max(
        len(base_model.layers) - n_layers,
        0,
    )

    trainable = 0

    for i, layer in enumerate(
        base_model.layers
    ):

        is_bn = isinstance(
            layer,
            layers.BatchNormalization,
        )

        layer.trainable = (
            i >= cutoff
            and not is_bn
        )

        trainable += int(
            layer.trainable
        )

    print(
        f"  Unfroze {trainable} layer(s) "
        f"in the top {n_layers} layers."
    )


# ============================================================
# CALLBACKS
# ============================================================

def make_callbacks(
    keras,
    checkpoint_path,
):

    return [

        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),

    ]


# ============================================================
# CLASS NAME FILES
# ============================================================

def save_class_names(
    class_names,
):

    os.makedirs(
        "models",
        exist_ok=True,
    )

    with open(
        CLASS_NAMES_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            class_names,
            handle,
            indent=2,
        )

    with open(
        CLASS_INDICES_PATH,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            {
                name: i
                for i, name in enumerate(
                    class_names
                )
            },
            handle,
            indent=2,
        )

    print(
        f"\nClass order saved:"
    )

    print(
        f"  {CLASS_NAMES_PATH}"
    )

    print(
        f"  {CLASS_INDICES_PATH}"
    )

    print(
        f"  {class_names}"
    )


# ============================================================
# MODEL CLASS CHECK
# ============================================================

def verify_loaded_model(
    model,
    class_names,
):

    output_shape = model.output_shape

    if isinstance(
        output_shape,
        list,
    ):

        output_shape = output_shape[0]

    model_classes = output_shape[-1]

    expected_classes = len(
        class_names
    )

    print(
        "\nExisting model output:"
    )

    print(
        f"  Model classes : {model_classes}"
    )

    print(
        f"  Dataset classes: {expected_classes}"
    )

    if model_classes != expected_classes:

        raise RuntimeError(
            "\nCLASS COUNT MISMATCH!\n\n"
            f"Existing model outputs "
            f"{model_classes} classes, but "
            f"the dataset contains "
            f"{expected_classes} classes.\n\n"
            f"Dataset classes:\n"
            f"{class_names}\n\n"
            "Do NOT resume this model "
            "with this dataset."
        )

    print(
        "  Class count check: OK"
    )


# ============================================================
# EVALUATION
# ============================================================

def report_predictions(
    model,
    val_dataset,
    val_items,
    class_names,
):

    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
    )

    y_true = [
        label
        for _, label in val_items
    ]

    y_pred = (
        model.predict(
            val_dataset,
            verbose=0,
        )
        .argmax(axis=1)
    )

    if len(y_pred) != len(y_true):

        print(
            "\nSkipping per-class report:"
        )

        print(
            f"Predictions: {len(y_pred)}"
        )

        print(
            f"Validation images: {len(y_true)}"
        )

        return

    labels = list(
        range(len(class_names))
    )

    print(
        "\nPer-class report:\n"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(rows = true, columns = predicted):"
    )

    print(
        "Class order:",
        class_names,
    )

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train/fine-tune a "
            "MobileNetV2 pet-emotion classifier"
        )
    )

    parser.add_argument(
        "--dataset",
        default=DATASET_DIR,
    )

    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Path to an existing .keras "
            "model to fine-tune"
        ),
    )

    parser.add_argument(
        "--head-epochs",
        type=int,
        default=HEAD_EPOCHS,
    )

    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=FINETUNE_EPOCHS,
    )

    parser.add_argument(
        "--unfreeze-layers",
        type=int,
        default=UNFREEZE_LAYERS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=FINETUNE_LR,
    )

    parser.add_argument(
        "--quarantine",
        action="store_true",
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
    )

    parser.add_argument(
        "--verify-pipeline",
        action="store_true",
    )

    args = parser.parse_args()

    dataset_dir = args.dataset

    train_dir = os.path.join(
        dataset_dir,
        "train",
    )

    val_dir = os.path.join(
        dataset_dir,
        "val",
    )

    if not os.path.exists(
        dataset_dir
    ):

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"{dataset_dir}"
        )

    # ========================================================
    # CLASS DISCOVERY
    # ========================================================

    class_names = find_class_names(
        train_dir,
        val_dir,
    )

    num_classes = len(
        class_names
    )

    print(
        "\nClasses found:"
    )

    for i, name in enumerate(
        class_names
    ):

        print(
            f"  {i}: {name}"
        )

    # ========================================================
    # FIND TRAINING IMAGES
    # ========================================================

    train_items, train_skipped = (
        find_image_files(
            train_dir,
            class_names,
        )
    )

    val_items, val_skipped = (
        find_image_files(
            val_dir,
            class_names,
        )
    )

    if not train_items:

        raise RuntimeError(
            "No training images found."
        )

    if not val_items:

        raise RuntimeError(
            "No validation images found."
        )

    print(
        f"\nFound:"
    )

    print(
        f"  Training images:   "
        f"{len(train_items)}"
    )

    print(
        f"  Validation images: "
        f"{len(val_items)}"
    )

    # ========================================================
    # CORRUPTION SCAN
    # ========================================================

    print(
        "\nScanning training images..."
    )

    good_train, bad_train = (
        scan_dataset(
            train_items
        )
    )

    report_scan(
        good_train,
        bad_train,
        class_names,
        train_skipped,
        "TRAIN",
    )

    print(
        "\nScanning validation images..."
    )

    good_val, bad_val = (
        scan_dataset(
            val_items
        )
    )

    report_scan(
        good_val,
        bad_val,
        class_names,
        val_skipped,
        "VALIDATION",
    )

    all_bad = (
        bad_train + bad_val
    )

    if all_bad:

        write_corrupt_report(
            all_bad,
            dataset_dir,
            CORRUPT_REPORT_PATH,
        )

        print(
            f"\nCorrupted-file report:"
            f" {CORRUPT_REPORT_PATH}"
        )

        if args.quarantine:

            moved_train = quarantine(
                bad_train,
                train_dir,
                os.path.join(
                    QUARANTINE_DIR,
                    "train",
                ),
            )

            moved_val = quarantine(
                bad_val,
                val_dir,
                os.path.join(
                    QUARANTINE_DIR,
                    "val",
                ),
            )

            print(
                f"Moved {moved_train + moved_val} "
                f"corrupted files."
            )

    if args.scan_only:

        print(
            "\n--scan-only enabled. "
            "Stopping."
        )

        return 0

    # Use only successfully scanned files.
    train_items = good_train
    val_items = good_val

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_weights = make_class_weights(
        train_items,
        num_classes,
    )

    print(
        "\nClass weights:"
    )

    for i, name in enumerate(
        class_names
    ):

        print(
            f"  {name:30s}"
            f"{class_weights.get(i, 0.0):.3f}"
        )

    # ========================================================
    # TENSORFLOW
    # ========================================================

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print(
        "\n" + "=" * 60
    )

    print(
        "TensorFlow:",
        tf.__version__,
    )

    print(
        "=" * 60
    )

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    if gpus:

        print(
            "\nGPU detected:"
        )

        for gpu in gpus:

            print(
                " ",
                gpu,
            )

        try:

            for gpu in gpus:

                tf.config.experimental.set_memory_growth(
                    gpu,
                    True,
                )

            print(
                "GPU memory growth enabled."
            )

        except RuntimeError as exc:

            print(
                "Could not enable "
                "GPU memory growth:",
                exc,
            )

    else:

        print(
            "\nNo GPU detected. "
            "Training will use CPU."
        )

    # ========================================================
    # DATASETS
    # ========================================================

    print(
        "\nBuilding datasets..."
    )

    train_dataset = build_dataset(
        tf,
        train_items,
        num_classes,
        args.batch_size,
        shuffle=True,
        seed=SEED,
        img_size=IMG_SIZE,
    )

    val_dataset = build_dataset(
        tf,
        val_items,
        num_classes,
        args.batch_size,
        shuffle=False,
        seed=SEED,
        img_size=IMG_SIZE,
    )

    if args.verify_pipeline:

        print(
            "\nChecking validation pipeline..."
        )

        survived = count_elements(
            val_dataset
        )

        if survived != len(val_items):

            print(
                f"WARNING: "
                f"{len(val_items) - survived} "
                f"validation image(s) were dropped."
            )

        else:

            print(
                f"OK - all {survived} "
                f"validation images survived."
            )

    # ========================================================
    # SAVE CLASS ORDER
    # ========================================================

    save_class_names(
        class_names
    )

    # ========================================================
    # MODEL
    # ========================================================

    if args.resume:

        # ====================================================
        # RESUME EXISTING MODEL
        # ====================================================

        if not os.path.isfile(
            args.resume
        ):

            raise FileNotFoundError(
                f"\nResume model not found:\n"
                f"{args.resume}"
            )

        print(
            "\n" + "=" * 60
        )

        print(
            "RESUMING EXISTING MODEL"
        )

        print(
            "=" * 60
        )

        print(
            f"\nLoading:\n"
            f"  {args.resume}"
        )

        model = keras.models.load_model(
            args.resume,
            compile=False,
        )

        verify_loaded_model(
            model,
            class_names,
        )

        # Find MobileNetV2 inside the loaded model.
        base_model = None

        for layer in model.layers:

            if isinstance(
                layer,
                keras.Model,
            ):

                if (
                    "mobilenetv2"
                    in layer.name.lower()
                ):

                    base_model = layer
                    break

        if base_model is None:

            # Search deeper by layer name.
            for layer in model.layers:

                if (
                    "mobilenet"
                    in layer.name.lower()
                ):

                    base_model = layer
                    break

        if base_model is None:

            raise RuntimeError(
                "\nCould not locate the "
                "MobileNetV2 backbone inside "
                "the loaded model.\n\n"
                "The resume model should be "
                "the model produced by this "
                "training script."
            )

        print(
            f"\nBackbone found:"
            f" {base_model.name}"
        )

        # IMPORTANT:
        # Start with the backbone frozen,
        # then explicitly unfreeze the top layers.
        base_model.trainable = True

        cutoff = max(
            len(base_model.layers)
            - args.unfreeze_layers,
            0,
        )

        trainable = 0

        for i, layer in enumerate(
            base_model.layers
        ):

            is_bn = isinstance(
                layer,
                layers.BatchNormalization,
            )

            layer.trainable = (
                i >= cutoff
                and not is_bn
            )

            trainable += int(
                layer.trainable
            )

        print(
            f"Unfroze {trainable} "
            f"backbone layers."
        )

        compile_model(
            keras,
            model,
            args.finetune_lr,
        )

        print(
            "\nFine-tuning existing model:"
        )

        print(
            f"  Epochs: "
            f"{args.finetune_epochs}"
        )

        print(
            f"  Learning rate: "
            f"{args.finetune_lr}"
        )

        print(
            f"  Unfrozen layers: "
            f"{args.unfreeze_layers}"
        )

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.finetune_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(
                keras,
                MODEL_PATH,
            ),
        )

        best_accuracy = max(
            history.history[
                "val_accuracy"
            ]
        )

        print(
            "\nBest resumed-model "
            f"val_accuracy: "
            f"{best_accuracy:.4f}"
        )

    else:

        # ====================================================
        # NEW MODEL
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "BUILDING NEW MOBILE NET V2 MODEL"
        )

        print(
            "=" * 60
        )

        model, base_model = build_model(
            tf,
            keras,
            layers,
            num_classes,
            IMG_SIZE,
            HEAD_LR,
        )

        model.summary()

        # ====================================================
        # PHASE 1
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            f"PHASE 1: HEAD ONLY"
        )

        print(
            f"Epochs: {args.head_epochs}"
        )

        print(
            f"Learning rate: {HEAD_LR}"
        )

        print(
            "=" * 60
        )

        history_head = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.head_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(
                keras,
                HEAD_MODEL_PATH,
            ),
        )

        head_best = max(
            history_head.history[
                "val_accuracy"
            ]
        )

        # ====================================================
        # PHASE 2
        # ====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            f"PHASE 2: FINE-TUNING"
        )

        print(
            f"Top layers: "
            f"{args.unfreeze_layers}"
        )

        print(
            f"Epochs: "
            f"{args.finetune_epochs}"
        )

        print(
            f"Learning rate: "
            f"{FINETUNE_LR}"
        )

        print(
            "=" * 60
        )

        unfreeze_top_layers(
            layers,
            base_model,
            args.unfreeze_layers,
        )

        compile_model(
            keras,
            model,
            FINETUNE_LR,
        )

        history_ft = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.finetune_epochs,
            class_weight=class_weights,
            callbacks=make_callbacks(
                keras,
                MODEL_PATH,
            ),
        )

        ft_best = max(
            history_ft.history[
                "val_accuracy"
            ]
        )

        print(
            "\nBest phase-1 "
            f"val_accuracy: {head_best:.4f}"
        )

        print(
            "Best phase-2 "
            f"val_accuracy: {ft_best:.4f}"
        )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 60
    )

    results = model.evaluate(
        val_dataset,
        verbose=1,
    )

    print(
        "\nFinal validation results:"
    )

    for name, value in zip(
        model.metrics_names,
        results,
    ):

        print(
            f"  {name}: {value:.4f}"
        )

    report_predictions(
        model,
        val_dataset,
        val_items,
        class_names,
    )

    if all_bad:

        print(
            f"\nWARNING:"
        )

        print(
            f"{len(all_bad)} corrupted "
            f"file(s) were excluded."
        )

        print(
            f"See {CORRUPT_REPORT_PATH}"
        )

    print(
        "\nTraining complete."
    )

    print(
        f"Best model/checkpoint:"
        f" {MODEL_PATH}"
    )

    print(
        f"Class names:"
        f" {CLASS_NAMES_PATH}"
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )

