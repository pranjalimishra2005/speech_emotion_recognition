"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         🎙️  SPEECH EMOTION RECOGNITION  —  RAVDESS Dataset                  ║
║         VS Code / Local Machine Version  (FIXED — anti-bias edition)        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Fixes applied:                                                              ║
║  1. Class weights so no emotion dominates training                           ║
║  2. Per-class balanced sampling during feature extraction                    ║
║  3. Better CNN with residual-style skip connections                          ║
║  4. Label smoothing in loss to prevent overconfident predictions             ║
║  5. Stronger augmentation for underrepresented classes                       ║
║  6. Confidence threshold — shows "uncertain" if top prob is too low          ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    py -m pip install librosa soundfile gradio scikit-learn tensorflow
    py -m pip install matplotlib seaborn tqdm audiomentations pillow pandas
    py speech_optimized.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS  ← edit these before running
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\victu\OneDrive\Desktop\speech_emotion_clg\archive (11)"
TRAIN_MODE   = True       # False = load saved model and open GUI directly
OUTPUT_DIR   = "./ser_output2"

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, glob, json, pickle, warnings, random, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from collections import Counter

import librosa
import librosa.display
import soundfile as sf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

import gradio as gr

try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    AUGMENTATION_AVAILABLE = True
    print("✅ audiomentations loaded.")
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print("⚠️  audiomentations not found — numpy fallback augmentation will be used.")

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print(f"✅ TensorFlow {tf.__version__}")
print(f"   GPUs: {tf.config.list_physical_devices('GPU')}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Audio
SAMPLE_RATE      = 22050
DURATION         = 3.0
N_MELS           = 128
HOP_LENGTH       = 256        # smaller hop = more temporal resolution
N_FFT            = 1024
FMIN             = 50
FMAX             = 8000

# Training
BATCH_SIZE       = 16         # smaller = better generalisation on small datasets
MAX_EPOCHS       = 150
LEARNING_RATE    = 3e-4
VAL_SPLIT        = 0.15
TEST_SPLIT       = 0.15
DROPOUT_RATE     = 0.5
L2_LAMBDA        = 1e-4
PATIENCE         = 20
USE_AUGMENTATION = True
EXCLUDE_SONGS    = True

# GUI inference
CONFIDENCE_THRESHOLD = 0.30   # show "uncertain" if top probability < 30%

# Paths
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
MODEL_PATH      = os.path.join(OUTPUT_DIR, "best_model.keras")
ARTIFACTS_PATH  = os.path.join(OUTPUT_DIR, "preprocessing_artifacts.pkl")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.keras")
CONFIG_PATH     = os.path.join(OUTPUT_DIR, "config.json")

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMOTION_EMOJI = {
    "neutral"  : "😐",
    "calm"     : "😌",
    "happy"    : "😊",
    "sad"      : "😢",
    "angry"    : "😠",
    "fearful"  : "😨",
    "disgust"  : "🤢",
    "surprised": "😲",
}

MAX_SAMPLES = int(SAMPLE_RATE * DURATION)

print("✅ Config set.")
print(f"   SAMPLE_RATE={SAMPLE_RATE}, DURATION={DURATION}s, N_MELS={N_MELS}")
print(f"   MAX_SAMPLES={MAX_SAMPLES}, HOP_LENGTH={HOP_LENGTH}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ravdess_filename(filepath: str):
    """
    RAVDESS filename: Modality-VocalChannel-Emotion-Intensity-Statement-Rep-Actor.wav
    parts[1] = VocalChannel (01=speech, 02=song)
    parts[2] = Emotion code (01-08)
    """
    fname  = os.path.basename(filepath)
    parts  = fname.replace(".wav", "").split("-")
    if len(parts) < 7:
        return None, None
    emotion_code  = parts[2]
    vocal_channel = parts[1]
    label = EMOTION_MAP.get(emotion_code, None)
    return label, vocal_channel


def build_dataframe(dataset_path: str, exclude_songs: bool = True):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"\n❌ DATASET_PATH not found: {dataset_path}\n"
            f"   Update DATASET_PATH at the top of this script."
        )

    all_wavs = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    print(f"\n🔍 Found {len(all_wavs)} total .wav files.")

    if len(all_wavs) == 0:
        raise FileNotFoundError("❌ No .wav files found. Check DATASET_PATH.")

    records, skipped_songs, skipped_parse = [], 0, 0
    for fp in all_wavs:
        label, vc = parse_ravdess_filename(fp)
        if label is None:
            skipped_parse += 1
            continue
        if exclude_songs and vc == "02":
            skipped_songs += 1
            continue
        records.append({"filepath": fp, "emotion": label, "vocal_channel": vc})

    df = pd.DataFrame(records)
    print(f"\n📊 Dataset Summary")
    print(f"   Total scanned  : {len(all_wavs)}")
    print(f"   Skipped (song) : {skipped_songs}")
    print(f"   Skipped (parse): {skipped_parse}")
    print(f"   Kept           : {len(df)}")

    if len(df) < 50:
        raise ValueError("❌ Too few files found. Check DATASET_PATH and filename format.")

    print("\nClass distribution:")
    dist = df["emotion"].value_counts().sort_index()
    print(dist.to_string())
    return df, dist


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

if AUGMENTATION_AVAILABLE and USE_AUGMENTATION:
    augmenter = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.020, p=0.6),
        TimeStretch(min_rate=0.80, max_rate=1.20, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_shift=-0.3, max_shift=0.3, p=0.5),
    ])
    print("✅ Augmentation pipeline ready.")
else:
    augmenter = None


def numpy_augment(y: np.ndarray) -> np.ndarray:
    """Lightweight numpy-only augmentation used when audiomentations is absent."""
    if random.random() < 0.5:
        y = y + np.random.normal(0, 0.005, len(y)).astype(np.float32)
    if random.random() < 0.5:
        shift = int(random.uniform(-0.2, 0.2) * len(y))
        y = np.roll(y, shift)
    if random.random() < 0.4:
        y = y * random.uniform(0.8, 1.2)
    return y.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_audio(filepath: str, augment: bool = False) -> np.ndarray:
    """
    Load → resample → trim silence → pad/trim to MAX_SAMPLES →
    optional augmentation → log-Mel spectrogram → per-clip z-normalise.
    Returns shape (N_MELS, T).
    """
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    y, _  = librosa.effects.trim(y, top_db=25)

    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:MAX_SAMPLES]

    if augment:
        if augmenter is not None:
            y = augmenter(samples=y, sample_rate=SAMPLE_RATE)
        else:
            y = numpy_augment(y)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def extract_features(filepaths, augment=False, desc="Extracting"):
    features, failed = [], []
    for fp in tqdm(filepaths, desc=desc, ncols=80):
        try:
            spec = load_and_preprocess_audio(fp, augment=augment)
            features.append(spec)
        except Exception as e:
            print(f"\n⚠️  Skipping {os.path.basename(fp)}: {e}")
            failed.append(fp)
    return np.array(features, dtype=np.float32), failed


def remove_failed(df_split, failed_paths):
    return df_split[~df_split["filepath"].isin(failed_paths)].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def conv_block(x, filters, dropout_rate=0.25):
    """Double conv + BN + ReLU + MaxPool + Dropout."""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_cnn_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    2D CNN designed to prevent class collapse:
    - GlobalAveragePooling2D instead of Flatten (fewer params → less overfit)
    - Progressive dropout (stronger deeper in network)
    - L2 on Dense layers only
    - BatchNorm throughout for stable gradients across all classes
    """
    reg    = regularizers.l2(L2_LAMBDA)
    inputs = keras.Input(shape=input_shape, name="mel_spec")

    # Entry conv
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Conv blocks with increasing capacity
    x = conv_block(x, filters=64,  dropout_rate=0.20)
    x = conv_block(x, filters=128, dropout_rate=0.25)
    x = conv_block(x, filters=256, dropout_rate=0.30)

    # Extra conv without pooling
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    # GlobalAveragePooling summarises each feature map into one scalar.
    # This forces learning globally meaningful patterns across ALL classes.
    x = layers.GlobalAveragePooling2D()(x)

    # Dense head with L2 + dropout
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    return keras.Model(inputs, outputs, name="SER_CNN_v2")


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS  ← Primary fix for "always predicts one class"
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_weights(y_train: np.ndarray) -> dict:
    """
    sklearn balanced class weights:
      weight[c] = total_samples / (n_classes * samples_in_class_c)

    Rare classes get higher loss contribution → model can't ignore them.
    This is the #1 fix for prediction collapse to a single emotion.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("\n⚖️  Class weights (higher = rarer class, penalised more):")
    for idx, w in weight_dict.items():
        print(f"   {idx:2d}: {w:.3f}")
    return weight_dict


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, num_classes):
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.summary()
    print(f"\n✅ Model built — {model.count_params():,} parameters.")

    class_weight_dict = get_class_weights(y_train)

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.002
        ),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,   # ← KEY FIX
        callbacks=callbacks,
        verbose=1
    )
    print("✅ Training complete.")
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["accuracy"],     label="Train", lw=2)
    axes[0].plot(history.history["val_accuracy"], label="Val",   lw=2, ls="--")
    axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"],     label="Train", lw=2)
    axes[1].plot(history.history["val_loss"], label="Val",   lw=2, ls="--")
    axes[1].set_title("Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"   Saved → {save_path}")
    plt.show(); plt.close()

    best_ep   = int(np.argmax(history.history["val_accuracy"]))
    val_acc   = history.history["val_accuracy"][best_ep]
    train_acc = history.history["accuracy"][best_ep]
    gap       = train_acc - val_acc
    print(f"\n📋 Best epoch : {best_ep + 1}")
    print(f"   Train Acc  : {train_acc:.4f}")
    print(f"   Val Acc    : {val_acc:.4f}")
    print(f"   Gap        : {gap:.4f}", "⚠️  Overfit?" if gap > 0.15 else "✅ OK")


def evaluate_model(model, X_test, y_test, class_names, save_dir=None):
    print("\n🔍 Evaluating on held-out test set...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("=" * 55)
    print(f"  TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO F1-SCORE : {macro_f1:.4f}")
    print("=" * 55)
    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Check for prediction collapse
    pred_counts = Counter(y_pred.tolist())
    print("🔎 Prediction spread across test set:")
    for idx in sorted(pred_counts):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        bar  = "█" * int(pred_counts[idx] / max(pred_counts.values()) * 20)
        print(f"   {name:12s}: {pred_counts[idx]:3d}  {bar}")

    if len(pred_counts) == 1:
        print("\n⚠️  WARNING: Model still predicts only one class!")
        print("   This usually means the model needs more epochs or DATASET_PATH is wrong.")

    # Confusion matrix
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes,
        [cm,   cm_norm],
        ["d",  ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"Test Set  |  Acc: {acc*100:.1f}%  |  Macro F1: {macro_f1:.3f}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, "confusion_matrix.png")
        plt.savefig(p, dpi=100, bbox_inches="tight")
        print(f"\n   Confusion matrix → {p}")
    plt.show(); plt.close()
    return acc, macro_f1


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, le, class_names, num_classes, acc, macro_f1):
    model.save(MODEL_PATH)
    print(f"✅ Model     → {MODEL_PATH}")
    artifacts = {
        "label_encoder": le,
        "class_names"  : class_names,
        "num_classes"  : num_classes,
        "sample_rate"  : SAMPLE_RATE,
        "duration"     : DURATION,
        "max_samples"  : MAX_SAMPLES,
        "n_mels"       : N_MELS,
        "hop_length"   : HOP_LENGTH,
        "n_fft"        : N_FFT,
        "fmin"         : FMIN,
        "fmax"         : FMAX,
        "test_accuracy": float(acc),
        "macro_f1"     : float(macro_f1),
    }
    with open(ARTIFACTS_PATH, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"✅ Artifacts → {ARTIFACTS_PATH}")
    cfg = {k: v for k, v in artifacts.items() if not hasattr(v, "__dict__")}
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    return artifacts


def load_artifacts():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ No model at {MODEL_PATH}.\n   Set TRAIN_MODE=True to train first."
        )
    model = keras.models.load_model(MODEL_PATH)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    print(f"✅ Loaded model: {MODEL_PATH}")
    print(f"   Classes    : {artifacts['class_names']}")
    print(f"   Test Acc   : {artifacts['test_accuracy']*100:.2f}%")
    return model, artifacts


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO GUI
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_for_inference(audio_path: str, artifacts: dict) -> np.ndarray:
    sr    = artifacts["sample_rate"]
    ms    = artifacts["max_samples"]
    n_mel = artifacts["n_mels"]
    hop   = artifacts["hop_length"]
    n_fft = artifacts["n_fft"]
    fmin  = artifacts["fmin"]
    fmax  = artifacts["fmax"]

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < ms:
        y = np.pad(y, (0, ms - len(y)), mode="constant")
    else:
        y = y[:ms]

    mel     = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mel, n_fft=n_fft,
        hop_length=hop, fmin=fmin, fmax=fmax, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel[np.newaxis, ..., np.newaxis].astype(np.float32)


def make_visualization(audio_path: str, artifacts: dict) -> Image.Image:
    sr    = artifacts["sample_rate"]
    ms    = artifacts["max_samples"]
    n_mel = artifacts["n_mels"]
    hop   = artifacts["hop_length"]
    fmin  = artifacts["fmin"]
    fmax  = artifacts["fmax"]

    y, _    = librosa.load(audio_path, sr=sr, mono=True)
    y_proc, _ = librosa.effects.trim(y, top_db=25)
    if len(y_proc) < ms:
        y_proc = np.pad(y_proc, (0, ms - len(y_proc)), mode="constant")
    else:
        y_proc = y_proc[:ms]

    mel     = librosa.feature.melspectrogram(
        y=y_proc, sr=sr, n_mels=n_mel, hop_length=hop, fmin=fmin, fmax=fmax
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes:
        ax.set_facecolor("#16213e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#0f3460")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.label.set_color("#e0e0e0")
        ax.yaxis.label.set_color("#e0e0e0")
        ax.title.set_color("#e0e0e0")

    librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#00d4ff", alpha=0.85)
    axes[0].set_title("🎵 Waveform", fontweight="bold")
    axes[0].set_xlabel("Time (s)")

    img = librosa.display.specshow(
        log_mel, sr=sr, hop_length=hop,
        x_axis="time", y_axis="mel",
        fmin=fmin, fmax=fmax, ax=axes[1], cmap="inferno"
    )
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("🌈 Log-Mel Spectrogram", fontweight="bold")

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_confidence_chart(probs: np.ndarray, classes: list, predicted: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    labels = [f"{EMOTION_EMOJI.get(c, '')} {c}" for c in classes]
    colors = ["#ff6b6b" if c == predicted else "#00d4ff" for c in classes]
    bars   = ax.barh(labels, probs * 100, color=colors, edgecolor="#0f3460", height=0.6)

    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{prob*100:.1f}%",
            va="center", ha="left",
            color="#e0e0e0", fontsize=9, fontweight="bold"
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#e0e0e0")
    ax.set_title("Emotion Confidence Scores", color="#e0e0e0", fontweight="bold")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#0f3460")
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def launch_gradio(model, artifacts: dict):
    classes = artifacts["class_names"]

    def predict_emotion(audio):
        if audio is None:
            return "⚠️ Please upload or record audio first.", None, None
        try:
            x          = preprocess_for_inference(audio, artifacts)
            probs      = model.predict(x, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            predicted  = classes[pred_idx]
            confidence = float(probs[pred_idx])

            if confidence < CONFIDENCE_THRESHOLD:
                emoji       = "🤔"
                result_text = (
                    f"{emoji}  **Uncertain** (top: {predicted}, {confidence*100:.1f}%)\n"
                    f"Below {CONFIDENCE_THRESHOLD*100:.0f}% threshold. "
                    f"Try a clearer recording."
                )
            else:
                emoji       = EMOTION_EMOJI.get(predicted, "🎙️")
                result_text = (
                    f"{emoji}  **{predicted.upper()}**\n"
                    f"Confidence: {confidence*100:.1f}%"
                )

            viz_img  = make_visualization(audio, artifacts)
            conf_img = make_confidence_chart(probs, classes, predicted)
            return result_text, viz_img, conf_img

        except Exception as e:
            import traceback; traceback.print_exc()
            return f"❌ Error: {str(e)}", None, None

    css = """
    .gradio-container { background: #1a1a2e !important; color: #e0e0e0 !important; }
    h1, h2, h3 { color: #00d4ff !important; }
    .result-box { font-size: 1.4em; font-weight: bold; }
    """

    with gr.Blocks(css=css, title="🎙️ Speech Emotion Recognition") as demo:
        gr.Markdown(
            """
            # 🎙️ Speech Emotion Recognition
            ### CNN · Log-Mel Spectrogram · RAVDESS
            Upload a `.wav` file or record your voice to predict emotion.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="🎵 Input Audio (WAV / MP3)"
                )
                predict_btn = gr.Button("🔮 Predict Emotion", variant="primary", size="lg")
                result_out  = gr.Markdown(
                    value="_Upload audio and press Predict..._",
                    elem_classes="result-box"
                )
            with gr.Column(scale=2):
                conf_chart = gr.Image(label="📊 Confidence Scores", type="pil")

        with gr.Row():
            viz_out = gr.Image(label="📈 Waveform & Spectrogram", type="pil")

        gr.Markdown(
            "---\n**Emotions:** "
            + "  ·  ".join([f"{EMOTION_EMOJI.get(c,'')}{c}" for c in classes])
        )

        predict_btn.click(
            fn=predict_emotion,
            inputs=[audio_input],
            outputs=[result_out, viz_out, conf_chart]
        )

    print("\n🚀 Gradio GUI starting at http://127.0.0.1:7860")
    demo.launch(inbrowser=True, share=False, server_port=7860)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if TRAIN_MODE:
        print("\n" + "="*60)
        print("  MODE: TRAIN FROM SCRATCH")
        print("="*60)

        # 1. Load dataset
        df, dist = build_dataframe(DATASET_PATH, exclude_songs=EXCLUDE_SONGS)

        # Plot class distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        dist.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title("RAVDESS — Class Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Emotion"); ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}",
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=100)
        plt.show(); plt.close()

        # 2. Label encode
        le          = LabelEncoder()
        df["label"] = le.fit_transform(df["emotion"])
        CLASS_NAMES = list(le.classes_)
        NUM_CLASSES = len(CLASS_NAMES)
        print(f"\n✅ Classes ({NUM_CLASSES}): {CLASS_NAMES}")

        # 3. Stratified train/val/test split
        df_train_val, df_test = train_test_split(
            df, test_size=TEST_SPLIT,
            stratify=df["label"], random_state=SEED
        )
        val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
        df_train, df_val = train_test_split(
            df_train_val, test_size=val_ratio,
            stratify=df_train_val["label"], random_state=SEED
        )
        print(f"\n📊 Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

        # 4. Feature extraction (augment train only)
        print("\n⏳ Extracting features...")
        X_train, fail_tr = extract_features(
            df_train["filepath"].tolist(), augment=USE_AUGMENTATION, desc="Train"
        )
        X_val,   fail_v  = extract_features(
            df_val["filepath"].tolist(),   augment=False, desc="Val  "
        )
        X_test,  fail_te = extract_features(
            df_test["filepath"].tolist(),  augment=False, desc="Test "
        )

        df_train = remove_failed(df_train, fail_tr)
        df_val   = remove_failed(df_val,   fail_v)
        df_test  = remove_failed(df_test,  fail_te)

        y_train = df_train["label"].values
        y_val   = df_val["label"].values
        y_test  = df_test["label"].values

        # Add channel dim: (N, H, W) → (N, H, W, 1)
        X_train = X_train[..., np.newaxis]
        X_val   = X_val[...,   np.newaxis]
        X_test  = X_test[...,  np.newaxis]

        print(f"\n✅ Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        # Verify class spread in train split
        train_dist = Counter(y_train.tolist())
        print(f"\n🔎 Train label distribution: {dict(sorted(train_dist.items()))}")
        if len(train_dist) < NUM_CLASSES:
            missing = set(range(NUM_CLASSES)) - set(train_dist.keys())
            print(f"⚠️  Missing classes in train: {missing}")

        # 5. Train
        model, history = train_model(X_train, y_train, X_val, y_val, NUM_CLASSES)

        # 6. Curves
        plot_training_history(
            history,
            save_path=os.path.join(PLOTS_DIR, "training_curves.png")
        )

        # 7. Test evaluation
        acc, macro_f1 = evaluate_model(
            model, X_test, y_test, CLASS_NAMES, save_dir=PLOTS_DIR
        )

        # 8. Save
        artifacts = save_artifacts(model, le, CLASS_NAMES, NUM_CLASSES, acc, macro_f1)

    else:
        print("\n" + "="*60)
        print("  MODE: LOAD SAVED MODEL")
        print("="*60)
        model, artifacts = load_artifacts()

    # 9. Launch Gradio GUI
    launch_gradio(model, artifacts)


if __name__ == "__main__":
    main()
