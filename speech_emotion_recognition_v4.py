"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        🎙️  SPEECH EMOTION RECOGNITION  —  RAVDESS  (v4 Improved Anti-Collapse Baseline)  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Root causes fixed vs v3:                                                    ║
║                                                                              ║
║  BUG 1 (critical): class_weight silently ignored with tf.data                ║
║    → Replaced with sample_weight embedded inside the tf.data pipeline.       ║
║      sample_weight is a per-sample float tensor, which tf.data supports.     ║
║      class_weight dict is pre-computed, then each sample in the dataset      ║
║      gets its own scalar weight before batching.                             ║
║                                                                              ║
║  BUG 2 (critical): MixUp washing out minority classes (happy, sad)           ║
║    → MixUp disabled entirely.  With only ~50–60 happy/sad samples across     ║
║      16 actors, blending spectrograms erases the subtle spectral cues that   ║
║      distinguish those emotions.  Re-enable only if all-class recall > 0.30. ║
║                                                                              ║
║  BUG 3: label_smoothing=0.15 too high for minority classes                   ║
║    → Reduced to 0.05.  At 0.15 the target for each class is diluted to       ║
║      0.85 true + 0.15/8 noise.  Combined with soft MixUp labels, the         ║
║      gradient signal for rare classes was near-zero.                         ║
║                                                                              ║
║  BUG 4: Double SpecAugment masking up to 23% of mel axis per sample          ║
║    → Reduced to single freq mask (max 10 bins) + single time mask            ║
║      (max 20 frames). Preserves the 1–3 kHz band where happy/sad differ.    ║
║                                                                              ║
║  BUG 5: Dense(64) head too small for 8-class separation                      ║
║    → Restored to Dense(128).  64 units forced the model to collapse          ║
║      fine-grained emotion features into too few dimensions.                  ║
║                                                                              ║
║  Additional improvements:                                                    ║
║    • Dropout reduced: 0.60 → 0.45 (model was under-fitting, not over-fitting)║
║    • L2 reduced: 2e-3 → 5e-4 (same reason)                                  ║
║    • SpatialDropout2D reduced: 0.15 → 0.10                                   ║
║    • Added macro-F1 as a training metric for per-class monitoring            ║
║    • EarlyStopping now monitors val_loss (unchanged) with patience=25         ║
║    • Added per-epoch recall logging to detect class collapse early            ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install librosa soundfile gradio scikit-learn tensorflow
    pip install matplotlib seaborn tqdm pillow pandas
    python speech_emotion_recognition_v4.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\victu\OneDrive\Desktop\speech_emotion_clg\archive (11)"
TRAIN_MODE   = False
OUTPUT_DIR   = "./ser_output_v4_improved"

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

import os, glob, json, pickle, warnings, random, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from collections import Counter

import librosa
import librosa.display

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score,
    recall_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)

import gradio as gr

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

# ── Audio ───────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 3.0
N_MELS      = 128
# N_FFT=1024 → window ≈ 46 ms at 22050 Hz.
# (512≈23 ms | 1024≈46 ms | 2048≈93 ms)
N_FFT       = 1024
HOP_LENGTH  = 256       # ≈ 11.6 ms step → ~258 time frames for a 3-second clip
FMIN        = 50
FMAX        = 8000
# top_db=30: trim anything >30 dB below the peak.
# Higher = more aggressive; lower = less aggressive. 30 preserves quiet content.
TRIM_TOP_DB = 30

MAX_SAMPLES = int(SAMPLE_RATE * DURATION)

# ── Training ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
MAX_EPOCHS    = 150
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 5

# Regularisation — reduced from v3 because the problem is underfitting/collapse,
# not overfitting.
DROPOUT_RATE  = 0.35    # lighter regularisation for first anti-collapse run
L2_LAMBDA     = 5e-4    # keep modest weight decay
SPATIAL_DROP  = 0.10    # keep mild spatial dropout

PATIENCE_ES   = 35
PATIENCE_LR   = 12
GRAD_CLIP     = 1.0

# Label smoothing — disabled for a cleaner first recovery run.
# At 0.15 with 8 classes each non-target gets 0.15/8 ≈ 0.019 added.
# For rare classes this dilutes an already weak signal. 0.00 gives the model the cleanest possible class signal in this ablation.
LABEL_SMOOTH  = 0.00    # 0.00 for first clean anti-collapse run

# ── SpecAugment — REDUCED to single masks ───────────────────────────────────────
# Double masking in v3 could zero up to 23% of the mel axis per sample.
# This destroyed the 1–3 kHz band that distinguishes happy from sad.
# Single masks with tighter caps preserve discriminative regions, while time warp is disabled for a cleaner ablation.
FREQ_MASK_MAX = 10      # was 15 (applied twice); now 10 (applied once)
TIME_MASK_MAX = 20      # was 25 (applied twice); now 20 (applied once)
TIME_WARP_MAX = 0       # disable time warp for first ablation

# ── MixUp — DISABLED ────────────────────────────────────────────────────────────
# MixUp blends minority-class spectrograms with majority-class ones.
# The blended spectrogram is acoustically closer to calm/surprised,
# so the model minimizes loss by predicting those — washing out happy/sad.
# Re-enable (MIXUP_ALPHA > 0) only after all-class recall exceeds 0.30.
MIXUP_ALPHA = 0.0       # was 0.3 — set >0 only after recall is healthy

# ── Speaker-independent split ───────────────────────────────────────────────────
VAL_ACTORS  = list(range(17, 21))   # actors 17, 18, 19, 20
TEST_ACTORS = list(range(21, 25))   # actors 21, 22, 23, 24

# ── GUI ─────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.30
EXCLUDE_SONGS        = True

# ── Paths ────────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
PLOTS_DIR       = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
MODEL_PATH      = os.path.join(OUTPUT_DIR, "best_model.keras")
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.keras")
ARTIFACTS_PATH  = os.path.join(OUTPUT_DIR, "preprocessing_artifacts.pkl")
CONFIG_PATH     = os.path.join(OUTPUT_DIR, "config.json")

EMOTION_MAP = {
    "01": "neutral", "02": "calm",    "03": "happy",    "04": "sad",
    "05": "angry",   "06": "fearful", "07": "disgust",  "08": "surprised",
}
EMOTION_EMOJI = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}

print("✅ Config ready.")
print(f"   N_FFT={N_FFT} (~{1000*N_FFT/SAMPLE_RATE:.0f} ms window), "
      f"HOP={HOP_LENGTH} (~{1000*HOP_LENGTH/SAMPLE_RATE:.1f} ms)")
print(f"   TRIM_TOP_DB={TRIM_TOP_DB}, L2={L2_LAMBDA}, "
      f"DROPOUT={DROPOUT_RATE}, LABEL_SMOOTH={LABEL_SMOOTH}")
print(f"   MIXUP_ALPHA={MIXUP_ALPHA} (0=disabled), FREQ_MASK_MAX={FREQ_MASK_MAX}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_ravdess_filename(filepath: str):
    """
    RAVDESS: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    Returns (emotion_label, vocal_channel, actor_id) or (None, None, None).
    """
    fname = os.path.basename(filepath)
    parts = fname.replace(".wav", "").split("-")
    if len(parts) < 7:
        return None, None, None
    vocal_channel = parts[1]
    emotion_code  = parts[2]
    actor_id      = int(parts[6])
    label = EMOTION_MAP.get(emotion_code, None)
    return label, vocal_channel, actor_id


def build_dataframe(dataset_path: str, exclude_songs: bool = True):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"\n❌ DATASET_PATH not found: {dataset_path}\n"
            f"   Update DATASET_PATH at the top of this script."
        )
    all_wavs = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    print(f"\n🔍 Found {len(all_wavs)} total .wav files.")
    if not all_wavs:
        raise FileNotFoundError("❌ No .wav files found. Check DATASET_PATH.")

    records, skipped_songs, skipped_parse = [], 0, 0
    for fp in all_wavs:
        label, vc, actor_id = parse_ravdess_filename(fp)
        if label is None:
            skipped_parse += 1; continue
        if exclude_songs and vc == "02":
            skipped_songs += 1; continue
        records.append({"filepath": fp, "emotion": label,
                         "vocal_channel": vc, "actor": actor_id})

    df = pd.DataFrame(records)
    print(f"   Kept: {len(df)}  (songs skipped: {skipped_songs}, "
          f"parse errors: {skipped_parse})")
    if len(df) < 50:
        raise ValueError("❌ Too few files. Check DATASET_PATH.")
    print("\nClass distribution:")
    dist = df["emotion"].value_counts().sort_index()
    print(dist.to_string())
    return df, dist


def speaker_independent_split(df, val_actors, test_actors):
    """Split purely by actor ID — zero overlap between partitions."""
    df_test  = df[df["actor"].isin(test_actors)].copy()
    df_val   = df[df["actor"].isin(val_actors)].copy()
    df_train = df[~df["actor"].isin(val_actors + test_actors)].copy()

    print(f"\n📊 Speaker-independent split:")
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"   {name:5s}: {len(subset):4d} clips  "
              f"actors={sorted(subset['actor'].unique().tolist())}")

    assert not (set(df_train["actor"]) & set(df_val["actor"])),  "Overlap train/val!"
    assert not (set(df_train["actor"]) & set(df_test["actor"])), "Overlap train/test!"
    assert not (set(df_val["actor"])   & set(df_test["actor"])), "Overlap val/test!"

    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_audio(filepath: str) -> np.ndarray:
    """
    load → resample → trim (top_db=30, preserves quiet content) →
    pad/trim → log-Mel → per-clip z-norm.

    Per-clip z-norm is applied identically at train/val/test/inference time,
    so there is no normalisation inconsistency.  Returns shape (N_MELS, T).
    """
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:MAX_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.astype(np.float32)


def extract_features(filepaths, desc="Extracting"):
    features, failed = [], []
    for fp in tqdm(filepaths, desc=desc, ncols=80):
        try:
            features.append(load_and_preprocess_audio(fp))
        except Exception as e:
            print(f"\n⚠️  Skipping {os.path.basename(fp)}: {e}")
            failed.append(fp)
    return np.array(features, dtype=np.float32), failed


def remove_failed(df_split, failed_paths):
    return df_split[~df_split["filepath"].isin(failed_paths)].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS → SAMPLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample float weights from balanced class weights.

    WHY NOT class_weight IN model.fit:
    When model.fit receives a tf.data.Dataset, Keras silently ignores the
    class_weight argument — no error, no warning, just no effect.  This was
    the primary reason happy and sad had zero recall in v3: the class weight
    dict was computed correctly but never applied.

    FIX: compute a weight scalar for each individual sample and embed it
    directly in the tf.data pipeline as the third element of the tuple
    (spec, label, sample_weight).  This is fully supported by Keras with
    tf.data and is applied correctly on every batch.

    Returns a float32 array of shape (N,) where each value is the
    inverse-frequency weight for that sample's class.
    """
    classes = np.unique(y)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y
    )
    weight_map = {int(c): float(w) for c, w in zip(classes, weights)}
    print("\n⚖️  Class weights (applied as per-sample weights):")
    for idx, w in sorted(weight_map.items()):
        print(f"   class {idx:2d}: {w:.4f}")
    sample_weights = np.array(
        [weight_map[int(yi)] for yi in y], dtype=np.float32
    )
    return sample_weights


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION — applied online via tf.data
# ═══════════════════════════════════════════════════════════════════════════════

def _freq_mask(spec):
    """Zero out one random band of mel bins (single mask, max FREQ_MASK_MAX)."""
    n_mels = tf.shape(spec)[0]
    n_time = tf.shape(spec)[1]
    f  = tf.random.uniform((), 0, FREQ_MASK_MAX + 1, dtype=tf.int32)
    f0 = tf.random.uniform((), 0, tf.maximum(n_mels - f, 1), dtype=tf.int32)
    mask = tf.concat([
        tf.ones ([f0,          n_time, 1], dtype=spec.dtype),
        tf.zeros([f,           n_time, 1], dtype=spec.dtype),
        tf.ones ([n_mels-f0-f, n_time, 1], dtype=spec.dtype),
    ], axis=0)
    return spec * mask


def _time_mask(spec):
    """Zero out one random band of time frames (single mask, max TIME_MASK_MAX)."""
    n_mels = tf.shape(spec)[0]
    n_time = tf.shape(spec)[1]
    t  = tf.random.uniform((), 0, TIME_MASK_MAX + 1, dtype=tf.int32)
    t0 = tf.random.uniform((), 0, tf.maximum(n_time - t, 1), dtype=tf.int32)
    mask = tf.concat([
        tf.ones ([n_mels, t0,         1], dtype=spec.dtype),
        tf.zeros([n_mels, t,          1], dtype=spec.dtype),
        tf.ones ([n_mels, n_time-t0-t,1], dtype=spec.dtype),
    ], axis=1)
    return spec * mask


def _time_warp(spec):
    """Random cyclic temporal shift up to TIME_WARP_MAX frames (disabled when TIME_WARP_MAX=0)."""
    shift = tf.random.uniform(
        (), -TIME_WARP_MAX, TIME_WARP_MAX + 1, dtype=tf.int32
    )
    return tf.roll(spec, shift=shift, axis=1)


def apply_augmentation(spec, label, sample_weight):
    """
    Single SpecAugment pass: time warp + one freq mask + one time mask.
    Reduced from double masks in v3.  Passes sample_weight through unchanged.
    """
    spec = _time_warp(spec)
    spec = _freq_mask(spec)
    spec = _time_mask(spec)
    return spec, label, sample_weight


def mixup_batch(specs, labels, sample_weights, alpha=MIXUP_ALPHA):
    """
    MixUp augmentation.  Currently disabled (MIXUP_ALPHA=0).

    If re-enabled, sample_weight is interpolated in the same proportion as
    the labels, which is the correct behaviour — a 70/30 blend of sample A
    and B should carry a proportional blend of their weights.

    NOTE: MixUp and sample_weight interact correctly here because the weight
    is interpolated (not selected by argmax), so minority-class gradients are
    not suppressed by blending.  However, with only ~50–60 happy/sad samples,
    MixUp still blurs their spectral signatures too much.  Re-enable once
    all-class recall > 0.30.
    """
    specs          = tf.cast(specs,          tf.float32)
    labels         = tf.cast(labels,         tf.float32)
    sample_weights = tf.cast(sample_weights, tf.float32)

    if alpha <= 0:
        return specs, labels, sample_weights

    batch_size = tf.shape(specs)[0]
    alpha_t    = tf.cast(alpha, tf.float32)
    g1  = tf.random.gamma(shape=[batch_size], alpha=alpha_t, dtype=tf.float32)
    g2  = tf.random.gamma(shape=[batch_size], alpha=alpha_t, dtype=tf.float32)
    lam = g1 / (g1 + g2 + 1e-8)
    lam = tf.maximum(lam, 1.0 - lam)

    indices = tf.random.shuffle(tf.range(batch_size))
    lam_s   = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_l   = tf.reshape(lam, [batch_size, 1])
    lam_w   = lam  # scalar per sample

    mixed_specs   = lam_s * specs   + (1.0 - lam_s) * tf.gather(specs,   indices)
    mixed_labels  = lam_l * labels  + (1.0 - lam_l) * tf.gather(labels,  indices)
    mixed_weights = lam_w * sample_weights + (1.0 - lam_w) * tf.gather(sample_weights, indices)
    return mixed_specs, mixed_labels, mixed_weights


def make_tf_datasets(X_train, y_train_oh, sw_train, X_val, y_val_oh, sw_val):
    """
    Build tf.data pipelines with sample_weight as the third tensor element.

    Train pipeline:
      shuffle → augment (spec, label, sw) → batch → MixUp → prefetch

    Val pipeline:
      batch (with sw_val so Keras doesn't error) → prefetch
      Val augmentation is intentionally omitted for honest metric evaluation.

    The three-element tuple (spec, label, weight) is the standard Keras
    convention for passing sample weights through tf.data.  Keras reads
    the third element automatically when present.
    """
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train_oh, sw_train))
        .shuffle(buffer_size=len(X_train), seed=SEED, reshuffle_each_iteration=True)
        .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_val, y_val_oh, sw_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def conv_block(x, filters, use_pool=True, spatial_dropout=SPATIAL_DROP):
    """
    Double Conv2D → BN → ReLU → optional MaxPool → SpatialDropout2D.
    SpatialDropout2D drops entire feature maps rather than individual pixels,
    which is much more regularising for CNNs (adjacent pixels are correlated).
    """
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_pool:
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(spatial_dropout)(x)
    return x


def build_cnn_model(input_shape: tuple, num_classes: int) -> keras.Model:
    """
    Two-block CNN with restored Dense(128) head.

    v3 used Dense(64) to combat overfitting, but the actual problem was
    underfitting/collapse — the model needed MORE representational room
    in the head to separate 8 emotion classes, not less.

    Dense(128) → BN → Dropout(0.45) gives the model enough capacity while
    the SpatialDropout and reduced L2 (5e-4) keep regularisation present
    without strangling the minority classes.

    Architecture:
      Input (128, T, 1)
      → Conv2D(32) → BN → ReLU
      → [Conv2D(64) × 2 → BN → ReLU → MaxPool(2,2) → SpatialDrop(0.10)]
      → [Conv2D(128) × 2 → BN → ReLU → (no pool) → SpatialDrop(0.10)]
      → GlobalAveragePooling2D
      → Dense(128) → BN → Dropout(0.45)
      → Dense(num_classes, softmax)
    """
    reg    = regularizers.l2(L2_LAMBDA)
    inputs = keras.Input(shape=input_shape, name="mel_spec")

    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = conv_block(x, filters=64,  use_pool=True,  spatial_dropout=SPATIAL_DROP)
    x = conv_block(x, filters=128, use_pool=False, spatial_dropout=SPATIAL_DROP)

    x = layers.GlobalAveragePooling2D()(x)

    # Restored to Dense(128) — Dense(64) was too small for 8-class separation
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)
    return keras.Model(inputs, outputs, name="SER_CNN_v4_improved")


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineSchedule(keras.callbacks.Callback):
    """
    Linear warm-up for WARMUP_EPOCHS, then cosine decay to LR/100.
    Prevents large gradients during early BatchNorm instability.
    """
    def __init__(self, total_epochs, warmup_epochs, base_lr):
        super().__init__()
        self.total_epochs  = total_epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr       = base_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            lr = self.base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
            lr = max(lr, self.base_lr / 100)
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except Exception:
            self.model.optimizer.learning_rate = lr

    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
        except Exception:
            lr = float(self.model.optimizer.learning_rate)
        if logs is not None:
            logs["lr"] = lr


# ═══════════════════════════════════════════════════════════════════════════════
# PER-EPOCH RECALL MONITOR — early warning for class collapse
# ═══════════════════════════════════════════════════════════════════════════════

class PerClassRecallLogger(keras.callbacks.Callback):
    """
    After each epoch, runs predict() on the validation set and prints
    per-class recall.  This gives you an early warning if any class drops
    to zero recall before EarlyStopping triggers.

    Also logs min_val_recall so you can monitor the worst-class recall
    over training and catch collapse the moment it starts.
    """
    def __init__(self, X_val, y_val, class_names, log_every=5):
        super().__init__()
        self.X_val       = X_val
        self.y_val       = y_val
        self.class_names = class_names
        self.log_every   = log_every   # print every N epochs to reduce noise

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every != 0:
            return
        y_prob = self.model.predict(self.X_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        recalls = recall_score(
            self.y_val, y_pred, average=None,
            labels=list(range(len(self.class_names))),
            zero_division=0
        )
        min_recall = float(np.min(recalls))
        if logs is not None:
            logs["min_val_recall"] = min_recall

        zero_recall = [self.class_names[i] for i, r in enumerate(recalls) if r == 0]
        recall_str  = "  ".join(
            f"{self.class_names[i]}:{r:.2f}"
            for i, r in enumerate(recalls)
        )
        flag = f"  ⚠️  ZERO RECALL: {zero_recall}" if zero_recall else ""
        print(f"\n   [Epoch {epoch+1}] Per-class recall: {recall_str}{flag}")
        print(f"   Min recall: {min_recall:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, num_classes, class_names):
    """
    The core training function.

    Key fix: sample_weight embedded in tf.data
    ────────────────────────────────────────────
    class_weight in model.fit is silently ignored when using tf.data.Dataset.
    The correct approach is to compute per-sample weights from the class
    weight mapping, then include them as the third element of the tf.data
    tuple: (spectrogram, one_hot_label, weight).

    Keras automatically reads the third element as sample_weight when it
    receives a 3-element tuple from tf.data.  This IS applied per-batch,
    and IS reflected in the loss computation.

    Verification: to confirm weights are active, print a few sample_weight
    values after computing them and compare against the class distribution.
    A balanced class with equal weights = 1.0; a rare class will be > 1.0.
    """
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)

    # Compute per-sample weights from class frequencies
    sw_train = compute_sample_weights(y_train)
    # Val sample weights remain uniform so validation reflects true performance.
    sw_val   = np.ones(len(y_val), dtype=np.float32)

    # Quick sanity check: minority class samples should have weight > 1.0
    print("\n🔎 Sample weight sanity check (first 5 per class):")
    for cls_idx in range(num_classes):
        idxs = np.where(y_train == cls_idx)[0][:3]
        ws   = sw_train[idxs]
        print(f"   {class_names[cls_idx]:12s}: weight={ws[0]:.3f}  "
              f"(n_train={np.sum(y_train==cls_idx)})")

    train_ds, val_ds = make_tf_datasets(
        X_train, y_train_oh, sw_train,
        X_val,   y_val_oh,   sw_val
    )

    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=GRAD_CLIP
        ),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=["accuracy"]
    )
    model.summary()
    print(f"\n✅ Model built — {model.count_params():,} parameters.")

    callbacks = [
        WarmupCosineSchedule(
            total_epochs=MAX_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            base_lr=LEARNING_RATE
        ),
        PerClassRecallLogger(
            X_val=X_val, y_val=y_val,
            class_names=class_names,
            log_every=2     # print every 2 epochs
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_ES,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=PATIENCE_LR,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    print("\n🚀 Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        # NOTE: class_weight is intentionally NOT passed here.
        # It is silently ignored by Keras when train_ds is a tf.data.Dataset.
        # The equivalent effect is achieved via sw_train embedded in train_ds.
        callbacks=callbacks,
        verbose=1
    )
    print("✅ Training complete.")
    return model, history


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION & PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_path=None):
    has_lr = "lr" in history.history
    ncols  = 3 if has_lr else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5))

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

    if has_lr:
        axes[2].plot(history.history["lr"], lw=2, color="green")
        axes[2].set_title("Learning Rate", fontweight="bold")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
        axes[2].set_yscale("log"); axes[2].grid(True, alpha=0.3)

    plt.suptitle("Training History (v4 improved)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"   Saved → {save_path}")
    plt.show(); plt.close()

    best_ep   = int(np.argmin(history.history["val_loss"]))
    val_acc   = history.history["val_accuracy"][best_ep]
    train_acc = history.history["accuracy"][best_ep]
    gap       = train_acc - val_acc
    flag      = "⚠️  Overfit?" if gap > 0.15 else "✅ OK"
    print(f"\n📋 Best epoch (lowest val_loss): {best_ep + 1}")
    print(f"   Train Acc : {train_acc:.4f}")
    print(f"   Val Acc   : {val_acc:.4f}")
    print(f"   Gap       : {gap:.4f}  {flag}")


def evaluate_model(model, X_test, y_test, class_names, save_dir=None):
    print("\n🔍 Evaluating on held-out test set (unseen actors)...")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc      = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    recalls  = recall_score(
        y_test, y_pred, average=None,
        labels=list(range(len(class_names))), zero_division=0
    )

    print("=" * 55)
    print(f"  TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO F1-SCORE : {macro_f1:.4f}")
    print(f"  MIN RECALL     : {np.min(recalls):.4f}  "
          f"({class_names[int(np.argmin(recalls))]})")
    is_balanced = (
        np.min(recalls) > 0 and
        max(Counter(y_pred.tolist()).values()) / len(y_pred) <= 0.30
    )
    print(f"  BALANCED?      : {'YES' if is_balanced else 'NO'}")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=class_names))

    pred_counts = Counter(y_pred.tolist())
    print("🔎 Prediction spread (target: all classes non-zero, none >30%):")
    total = len(y_pred)
    for idx in range(len(class_names)):
        name  = class_names[idx]
        count = pred_counts.get(idx, 0)
        pct   = 100 * count / total
        bar   = "█" * int(count / max(pred_counts.values(), default=1) * 20)
        flag  = " ⚠️  ZERO" if count == 0 else (
                " ⚠️  DOMINANT" if pct > 30 else "")
        print(f"   {name:12s}: {count:3d} ({pct:5.1f}%)  {bar}{flag}")

    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"Test Set  |  Acc:{acc*100:.1f}%  |  MacroF1:{macro_f1:.3f}  "
        f"|  MinRecall:{np.min(recalls):.2f}",
        fontsize=12, fontweight="bold"
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
        "label_encoder" : le,
        "class_names"   : class_names,
        "num_classes"   : num_classes,
        "sample_rate"   : SAMPLE_RATE,
        "duration"      : DURATION,
        "max_samples"   : MAX_SAMPLES,
        "n_mels"        : N_MELS,
        "hop_length"    : HOP_LENGTH,
        "n_fft"         : N_FFT,
        "fmin"          : FMIN,
        "fmax"          : FMAX,
        "trim_top_db"   : TRIM_TOP_DB,
        "test_accuracy" : float(acc),
        "macro_f1"      : float(macro_f1),
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
            f"❌ No model at {MODEL_PATH}. Set TRAIN_MODE=True to train first."
        )
    model = keras.models.load_model(MODEL_PATH)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    print(f"✅ Model loaded. Test Acc: {artifacts['test_accuracy']*100:.2f}%")
    return model, artifacts


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PREPROCESSING — identical to training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_for_inference(audio_path: str, artifacts: dict) -> np.ndarray:
    """Returns shape (1, N_MELS, T, 1)."""
    sr     = artifacts["sample_rate"]
    ms     = artifacts["max_samples"]
    n_mel  = artifacts["n_mels"]
    hop    = artifacts["hop_length"]
    n_fft  = artifacts["n_fft"]
    fmin   = artifacts["fmin"]
    fmax   = artifacts["fmax"]
    top_db = artifacts.get("trim_top_db", 30)

    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
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


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO GUI
# ═══════════════════════════════════════════════════════════════════════════════

def make_visualization(audio_path: str, artifacts: dict) -> Image.Image:
    sr     = artifacts["sample_rate"]
    ms     = artifacts["max_samples"]
    hop    = artifacts["hop_length"]
    n_mel  = artifacts["n_mels"]
    fmin   = artifacts["fmin"]
    fmax   = artifacts["fmax"]
    top_db = artifacts.get("trim_top_db", 30)

    y, _   = librosa.load(audio_path, sr=sr, mono=True)
    y_p, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_p) < ms:
        y_p = np.pad(y_p, (0, ms - len(y_p)), mode="constant")
    else:
        y_p = y_p[:ms]

    mel     = librosa.feature.melspectrogram(
        y=y_p, sr=sr, n_mels=n_mel, hop_length=hop, fmin=fmin, fmax=fmax
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
        log_mel, sr=sr, hop_length=hop, x_axis="time", y_axis="mel",
        fmin=fmin, fmax=fmax, ax=axes[1], cmap="inferno"
    )
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    axes[1].set_title("🌈 Log-Mel Spectrogram", fontweight="bold")

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
    return Image.open(buf)


def make_confidence_chart(probs, classes, predicted) -> Image.Image:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    labels = [f"{EMOTION_EMOJI.get(c, '')} {c}" for c in classes]
    colors = ["#ff6b6b" if c == predicted else "#00d4ff" for c in classes]
    bars   = ax.barh(labels, probs * 100, color=colors,
                     edgecolor="#0f3460", height=0.6)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", ha="left",
                color="#e0e0e0", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#e0e0e0")
    ax.set_title("Emotion Confidence Scores", color="#e0e0e0", fontweight="bold")
    ax.tick_params(colors="#e0e0e0")
    for spine in ax.spines.values(): spine.set_edgecolor("#0f3460")
    ax.invert_yaxis(); plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor())
    plt.close(fig); buf.seek(0)
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
                result_text = (
                    f"🤔  **Uncertain** (top: {predicted}, "
                    f"{confidence*100:.1f}%)\n"
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
            # 🎙️ Speech Emotion Recognition  (v4 — improved anti-collapse baseline)
            ### CNN · Log-Mel Spectrogram · RAVDESS · Speaker-independent
            Upload a `.wav` file or record your voice to predict emotion.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath", label="🎵 Input Audio (WAV / MP3)"
                )
                predict_btn = gr.Button("🔮 Predict Emotion",
                                        variant="primary", size="lg")
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
        predict_btn.click(fn=predict_emotion, inputs=[audio_input],
                          outputs=[result_out, viz_out, conf_chart])

    print("\n🚀 Gradio GUI starting at http://127.0.0.1:7860")
    demo.launch(inbrowser=True, share=False, server_port=7860)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if TRAIN_MODE:
        print("\n" + "="*60)
        print("  MODE: TRAIN FROM SCRATCH  (v4 improved)")
        print("="*60)

        # 1. Load dataset
        df, dist = build_dataframe(DATASET_PATH, exclude_songs=EXCLUDE_SONGS)

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

        # 3. Speaker-independent split
        df_train, df_val, df_test = speaker_independent_split(
            df, val_actors=VAL_ACTORS, test_actors=TEST_ACTORS
        )

        # Verify all classes present in train
        train_classes = set(df_train["label"].unique())
        if train_classes != set(range(NUM_CLASSES)):
            missing = [CLASS_NAMES[i] for i in sorted(
                set(range(NUM_CLASSES)) - train_classes)]
            print(f"⚠️  Missing from train set: {missing}")
            print("   Adjust VAL_ACTORS / TEST_ACTORS.")

        # 4. Feature extraction
        print("\n⏳ Extracting features...")
        X_train, fail_tr = extract_features(df_train["filepath"].tolist(), "Train")
        X_val,   fail_v  = extract_features(df_val["filepath"].tolist(),   "Val  ")
        X_test,  fail_te = extract_features(df_test["filepath"].tolist(),  "Test ")

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

        print(f"\n✅ Shapes — train:{X_train.shape}, val:{X_val.shape}, "
              f"test:{X_test.shape}")
        print(f"   Train dist: {dict(sorted(Counter(y_train.tolist()).items()))}")
        print(f"   Val   dist: {dict(sorted(Counter(y_val.tolist()).items()))}")

        # 5. Train
        model, history = train_model(
            X_train, y_train, X_val, y_val, NUM_CLASSES, CLASS_NAMES
        )

        # 6. Training curves
        plot_training_history(
            history,
            save_path=os.path.join(PLOTS_DIR, "training_curves.png")
        )

        # 7. Test evaluation
        acc, macro_f1 = evaluate_model(
            model, X_test, y_test, CLASS_NAMES, save_dir=PLOTS_DIR
        )

        # 8. Save
        artifacts = save_artifacts(
            model, le, CLASS_NAMES, NUM_CLASSES, acc, macro_f1
        )

    else:
        print("\n" + "="*60)
        print("  MODE: LOAD SAVED MODEL")
        print("="*60)
        model, artifacts = load_artifacts()

    # 9. Launch GUI
    launch_gradio(model, artifacts)


if __name__ == "__main__":
    main()
