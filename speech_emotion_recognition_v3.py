"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        🎙️  SPEECH EMOTION RECOGNITION  —  RAVDESS  (v3 Optimized)           ║
║        Targeting reduced train-val gap on speaker-independent splits         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Changes from v2 → v3 (driven by observed curves):                          ║
║                                                                              ║
║  Architecture                                                                ║
║    • Removed second conv block's MaxPool → preserves more temporal info      ║
║    • Added SpatialDropout2D after each conv block (drops whole feature maps) ║
║    • Dense head shrunk: 128 → 64 units                                       ║
║    • Dropout raised: 0.50 → 0.60 on Dense layer                             ║
║    • L2 raised: 1e-3 → 2e-3                                                  ║
║                                                                              ║
║  Augmentation                                                                ║
║    • MixUp (α=0.3) applied per-batch in tf.data pipeline                    ║
║    • Double SpecAugment: 2 freq masks + 2 time masks per sample              ║
║    • Time-domain warp (random roll ±15 frames) added                         ║
║                                                                              ║
║  Training strategy                                                           ║
║    • Warm-up LR schedule: linear ramp for first 5 epochs, then cosine decay  ║
║    • Label smoothing raised: 0.1 → 0.15                                      ║
║    • Gradient clipping: clip_norm=1.0 to suppress loss spikes               ║
║    • EarlyStopping patience raised: 25 → 30                                  ║
║    • Validation monitored: val_loss (unchanged from v2)                      ║
║                                                                              ║
║  All v2 decisions kept                                                       ║
║    • Speaker-independent split (actors 17-20 val, 21-24 test)                ║
║    • Per-clip z-normalisation (consistent train/val/test/inference)          ║
║    • N_FFT=1024 (~46 ms window)                                               ║
║    • top_db=30 (preserves quiet emotional content)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    pip install librosa soundfile gradio scikit-learn tensorflow
    pip install matplotlib seaborn tqdm pillow pandas
    python speech_emotion_recognition_v3.py
"""

# ═══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\victu\OneDrive\Desktop\speech_emotion_clg\archive (11)"
TRAIN_MODE   = True
OUTPUT_DIR   = "./ser_output_v3"

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
    classification_report, confusion_matrix, f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LambdaCallback
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
SAMPLE_RATE  = 22050
DURATION     = 3.0
N_MELS       = 128
# N_FFT=1024 → window ≈ 46 ms at 22050 Hz.
# (512≈23 ms | 1024≈46 ms | 2048≈93 ms)
# 46 ms captures both spectral detail and pitch resolution for speech emotion.
N_FFT        = 1024
HOP_LENGTH   = 256   # ≈ 11.6 ms step → ~258 time frames for a 3-second clip
FMIN         = 50
FMAX         = 8000
# top_db=30: trim audio more than 30 dB below the clip peak.
# Higher value = more aggressive trimming; lower = less aggressive.
# 30 keeps quiet emotional content (sad, calm, fear).
TRIM_TOP_DB  = 30

MAX_SAMPLES  = int(SAMPLE_RATE * DURATION)

# ── Training ────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
MAX_EPOCHS    = 150
LEARNING_RATE = 3e-4
WARMUP_EPOCHS = 5       # linear LR warm-up before cosine decay kicks in
DROPOUT_RATE  = 0.60    # raised from 0.50 in v2
L2_LAMBDA     = 2e-3    # raised from 1e-3 in v2
PATIENCE_ES   = 30      # raised from 25; val_loss on small set needs patience
PATIENCE_LR   = 12
GRAD_CLIP     = 1.0     # gradient norm clipping — suppresses loss spikes
LABEL_SMOOTH  = 0.15    # raised from 0.10

# ── SpecAugment (applied twice per sample) ──────────────────────────────────────
FREQ_MASK_MAX = 15   # max mel bands to zero (applied twice)
TIME_MASK_MAX = 25   # max time frames to zero (applied twice)
TIME_WARP_MAX = 15   # max frames to randomly roll (temporal shift)

# ── MixUp ───────────────────────────────────────────────────────────────────────
MIXUP_ALPHA = 0.3    # Beta(α,α) interpolation strength; 0 = no mixup

# ── Speaker-independent split ───────────────────────────────────────────────────
# RAVDESS has 24 actors. Hold-out by actor, not by file.
# 16 train actors / 4 val actors / 4 test actors.
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
print(f"   SAMPLE_RATE={SAMPLE_RATE}, DURATION={DURATION}s, N_MELS={N_MELS}")
print(f"   N_FFT={N_FFT} (~{1000*N_FFT/SAMPLE_RATE:.0f} ms), "
      f"HOP_LENGTH={HOP_LENGTH} (~{1000*HOP_LENGTH/SAMPLE_RATE:.1f} ms)")
print(f"   TRIM_TOP_DB={TRIM_TOP_DB} (higher = more aggressive trimming)")
print(f"   L2={L2_LAMBDA}, DROPOUT={DROPOUT_RATE}, LABEL_SMOOTH={LABEL_SMOOTH}")
print(f"   MIXUP_ALPHA={MIXUP_ALPHA}, GRAD_CLIP={GRAD_CLIP}")


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
    """
    Split purely by actor ID — zero actor overlap between partitions.
    This is the standard evaluation protocol for RAVDESS.
    It gives an honest estimate of generalisation to unseen speakers.
    """
    df_test  = df[df["actor"].isin(test_actors)].copy()
    df_val   = df[df["actor"].isin(val_actors)].copy()
    df_train = df[~df["actor"].isin(val_actors + test_actors)].copy()

    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"   {name:5s}: {len(subset):4d} clips  "
              f"actors={sorted(subset['actor'].unique().tolist())}")

    assert not (set(df_train["actor"]) & set(df_val["actor"])), \
        "Actor overlap train/val!"
    assert not (set(df_train["actor"]) & set(df_test["actor"])), \
        "Actor overlap train/test!"
    assert not (set(df_val["actor"]) & set(df_test["actor"])), \
        "Actor overlap val/test!"

    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_audio(filepath: str) -> np.ndarray:
    """
    load → resample → trim silence (top_db=30, less aggressive, preserves
    quiet content) → pad/trim to MAX_SAMPLES → log-Mel spectrogram →
    per-clip z-normalise.

    Per-clip z-norm is a deliberate design choice — it equalises loudness
    across clips so the model focuses on spectral shape, not recording level.
    The same transform is applied identically at all stages (train/val/test/
    inference), so there is no normalisation inconsistency.

    Returns shape (N_MELS, T).
    """
    y, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)

    if len(y) < MAX_SAMPLES:
        y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:MAX_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX,
        power=2.0
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
# AUGMENTATION  — applied online via tf.data (fresh random mask every epoch)
# ═══════════════════════════════════════════════════════════════════════════════

def _freq_mask(spec):
    """Zero out a random band of mel frequency bins."""
    n_mels = tf.shape(spec)[0]
    n_time = tf.shape(spec)[1]
    f  = tf.random.uniform((), 0, FREQ_MASK_MAX + 1, dtype=tf.int32)
    f0 = tf.random.uniform((), 0, tf.maximum(n_mels - f, 1), dtype=tf.int32)
    mask = tf.concat([
        tf.ones ([f0,           n_time, 1], dtype=spec.dtype),
        tf.zeros([f,            n_time, 1], dtype=spec.dtype),
        tf.ones ([n_mels-f0-f,  n_time, 1], dtype=spec.dtype),
    ], axis=0)
    return spec * mask


def _time_mask(spec):
    """Zero out a random band of time frames."""
    n_mels = tf.shape(spec)[0]
    n_time = tf.shape(spec)[1]
    t  = tf.random.uniform((), 0, TIME_MASK_MAX + 1, dtype=tf.int32)
    t0 = tf.random.uniform((), 0, tf.maximum(n_time - t, 1), dtype=tf.int32)
    mask = tf.concat([
        tf.ones ([n_mels, t0,          1], dtype=spec.dtype),
        tf.zeros([n_mels, t,           1], dtype=spec.dtype),
        tf.ones ([n_mels, n_time-t0-t, 1], dtype=spec.dtype),
    ], axis=1)
    return spec * mask


def _time_warp(spec):
    """
    Random temporal roll (cyclic shift) by up to TIME_WARP_MAX frames.
    Simulates slight timing variation without distorting the spectrogram shape.
    """
    shift = tf.random.uniform(
        (), -TIME_WARP_MAX, TIME_WARP_MAX + 1, dtype=tf.int32
    )
    return tf.roll(spec, shift=shift, axis=1)


def apply_augmentation(spec, label):
    """
    Double SpecAugment (2 freq masks + 2 time masks) + time warp.
    Applied per-sample inside tf.data.map — a fresh random transform
    is drawn for every sample on every epoch.
    """
    spec = _time_warp(spec)
    spec = _freq_mask(spec)
    spec = _freq_mask(spec)    # second independent freq mask
    spec = _time_mask(spec)
    spec = _time_mask(spec)    # second independent time mask
    return spec, label


def mixup_batch(specs, labels, alpha=MIXUP_ALPHA):
    """
    MixUp augmentation applied at the batch level.
    Interpolates pairs of spectrograms and their one-hot labels:
        x̃ = λ·xᵢ + (1-λ)·xⱼ,   ỹ = λ·yᵢ + (1-λ)·yⱼ
    λ ~ Beta(α, α).  With α=0.3 the distribution is strongly bimodal
    (most interpolations close to the original samples) but occasionally
    produces a genuine 50/50 blend, which forces the decision boundary
    to generalise across emotion pairs.

    Important implementation detail:
    all tensors are explicitly cast to float32 so tf.data tracing does not
    fail with float64/float32 dtype mismatches.
    """
    specs = tf.cast(specs, tf.float32)
    labels = tf.cast(labels, tf.float32)

    batch_size = tf.shape(specs)[0]

    # Sample λ from Beta(α, α) via two Gamma samples, all in float32.
    alpha_t = tf.cast(alpha, tf.float32)
    g1 = tf.random.gamma(shape=[batch_size], alpha=alpha_t, dtype=tf.float32)
    g2 = tf.random.gamma(shape=[batch_size], alpha=alpha_t, dtype=tf.float32)
    lam = g1 / (g1 + g2 + tf.constant(1e-8, dtype=tf.float32))
    lam = tf.maximum(lam, tf.constant(1.0, dtype=tf.float32) - lam)

    # Random shuffle indices for mixing partners
    indices = tf.random.shuffle(tf.range(batch_size))
    lam_s = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_l = tf.reshape(lam, [batch_size, 1])

    mixed_specs = lam_s * specs + (tf.constant(1.0, dtype=tf.float32) - lam_s) * tf.gather(specs, indices)
    mixed_labels = lam_l * labels + (tf.constant(1.0, dtype=tf.float32) - lam_l) * tf.gather(labels, indices)
    return mixed_specs, mixed_labels


def make_tf_datasets(X_train, y_train_oh, X_val, y_val_oh):
    """
    Build tf.data pipelines.
    Train: shuffle → per-sample SpecAugment → batch → MixUp → prefetch.
    Val:   batch → prefetch (no augmentation — honest metric).
    """
    train_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train_oh))
        .shuffle(buffer_size=len(X_train), seed=SEED, reshuffle_each_iteration=True)
        .map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)   # drop_remainder needed for MixUp
        .map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset
        .from_tensor_slices((X_val, y_val_oh))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

def conv_block(x, filters, use_pool=True, spatial_dropout=0.15):
    """
    Double Conv → BN → ReLU → optional MaxPool → SpatialDropout2D.

    SpatialDropout2D drops entire feature maps (all spatial positions of a
    channel together).  This is much more effective than standard Dropout for
    convolutional layers because adjacent pixels are correlated — regular
    Dropout rarely zeroes out a full feature map and so provides weaker
    regularisation.  Rate 0.15 drops ~15% of feature maps per forward pass.

    use_pool=False on the second block preserves temporal resolution at the
    cost of a slightly larger feature map going into GlobalAveragePooling.
    This gives the pooling layer more temporal evidence to aggregate.
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
    Capacity-controlled 2D CNN for speaker-independent RAVDESS.

    Design rationale for v3:
    ─────────────────────────────────────────────────────────────
    • Two conv blocks only (64 → 128).  At ~800 training clips
      across 16 actors, three blocks overfit even with regularisation.

    • Block 1 uses MaxPool (2×2): fast spatial compression on mel axis.
    • Block 2 uses NO pool: preserves temporal resolution into GAP.
      More time frames averaged = more stable, less actor-specific features.

    • SpatialDropout2D(0.15) on both blocks: drops entire feature maps,
      preventing any single map from becoming a speaker-identity detector.

    • GlobalAveragePooling2D: one scalar per map → strong compression,
      no spatial memorisation.

    • Single Dense(64) head: deliberately tiny.  128 units in v2 could
      still memorise the 16-actor training set.  64 forces coarser,
      more generalisable representations.

    • Dropout(0.60) on Dense: very aggressive — about 60% of activations
      zeroed per forward pass, preventing the dense layer from memorising
      specific training examples.

    • L2=2e-3 on Dense: weight magnitude penalised throughout.

    Expected parameter count: ~500 k (vs ~1–2 M in v2).
    """
    reg    = regularizers.l2(L2_LAMBDA)
    inputs = keras.Input(shape=input_shape, name="mel_spec")

    # Entry conv (lightweight edge / harmonic detector)
    x = layers.Conv2D(32, (3, 3), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Block 1: pool on both axes → shrinks H and W
    x = conv_block(x, filters=64,  use_pool=True,  spatial_dropout=0.15)
    # Block 2: no pool → preserves temporal axis for GAP
    x = conv_block(x, filters=128, use_pool=False, spatial_dropout=0.15)

    # Global average pooling — averages all spatial positions per feature map
    x = layers.GlobalAveragePooling2D()(x)

    # Compact dense head
    x = layers.Dense(64, activation="relu", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)
    return keras.Model(inputs, outputs, name="SER_CNN_v3")


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE  — linear warm-up + cosine decay
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineSchedule(keras.callbacks.Callback):
    """
    Custom LR schedule:
      epochs 0 … WARMUP_EPOCHS-1 : LR ramps linearly from 0 → LEARNING_RATE
      epochs WARMUP_EPOCHS … end : LR follows cosine decay to LR/100

    Warm-up prevents very large gradients in the first few epochs when
    BatchNorm statistics are not yet stable, which was causing the early
    loss spikes visible in the v2 validation curve.

    Cosine decay keeps LR smoothly declining rather than step-wise, which
    tends to produce slightly better final minima.

    ReduceLROnPlateau is still present as a safety net for when val_loss
    stalls mid-training.
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
            lr = self.base_lr * 0.5 * (
                1.0 + np.cos(np.pi * progress)
            )
            lr = max(lr, self.base_lr / 100)   # floor at 1% of base LR
        # Keras/TensorFlow versions differ in how optimizer.learning_rate is stored
        # (variable, schedule wrapper, or plain value). Assign directly for compatibility.
        try:
            self.model.optimizer.learning_rate.assign(lr)
        except Exception:
            self.model.optimizer.learning_rate = lr

    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = float(self.model.optimizer.learning_rate)
        if logs is not None:
            logs["lr"] = lr


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def get_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    print("\n⚖️  Class weights:")
    for idx, w in weight_dict.items():
        print(f"   {idx:2d}: {w:.3f}")
    return weight_dict


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_val, y_val, num_classes):
    """
    Compile, build pipelines, and fit.

    Key decisions:
    ─────────────────────────────────────────────────────────────
    • CategoricalCrossentropy(label_smoothing=0.15): prevents the model
      from becoming overconfident, reducing the sharp loss spikes that
      appeared in v2 validation curves.

    • clip_norm=1.0: gradient clipping prevents outlier batches (which
      occur naturally when MixUp produces an unusual blend) from causing
      runaway parameter updates.

    • WarmupCosineSchedule takes precedence over ReduceLROnPlateau during
      warm-up; after warm-up both are active (cosine provides the primary
      schedule, plateau reduction is a safety net).

    • EarlyStopping monitors val_loss with patience=30.  val_loss is
      smoother than val_accuracy for small validation sets (~200 clips),
      and 30 epochs allows the cosine schedule to properly explore before
      committing to early exit.

    • class_weight applied to training to compensate for any speaker-
      induced class imbalance in the 16 training actors.
    """
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes)

    train_ds, val_ds = make_tf_datasets(X_train, y_train_oh, X_val, y_val_oh)

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

    class_weight_dict = get_class_weights(y_train)

    callbacks = [
        WarmupCosineSchedule(
            total_epochs=MAX_EPOCHS,
            warmup_epochs=WARMUP_EPOCHS,
            base_lr=LEARNING_RATE
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
        # Safety net: if val_loss plateaus despite cosine schedule, halve LR
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
        class_weight=class_weight_dict,
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

    ncols = 3 if has_lr else 2
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

    plt.suptitle("Training History (v3)", fontsize=14, fontweight="bold")
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

    print("=" * 55)
    print(f"  TEST ACCURACY  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  MACRO F1-SCORE : {macro_f1:.4f}")
    print("=" * 55)
    print(classification_report(y_test, y_pred, target_names=class_names))

    pred_counts = Counter(y_pred.tolist())
    print("🔎 Prediction spread:")
    for idx in sorted(pred_counts):
        name = class_names[idx] if idx < len(class_names) else str(idx)
        bar  = "█" * int(pred_counts[idx] / max(pred_counts.values()) * 20)
        print(f"   {name:12s}: {pred_counts[idx]:3d}  {bar}")

    if len(pred_counts) == 1:
        print("\n⚠️  Model predicts only one class — check class weights and data.")

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
            f"❌ No model at {MODEL_PATH}.\n   Set TRAIN_MODE=True to train first."
        )
    model = keras.models.load_model(MODEL_PATH)
    with open(ARTIFACTS_PATH, "rb") as f:
        artifacts = pickle.load(f)
    print(f"✅ Loaded model: {MODEL_PATH}")
    print(f"   Test Acc: {artifacts['test_accuracy']*100:.2f}%")
    return model, artifacts


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE PREPROCESSING  — identical pipeline to training
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_for_inference(audio_path: str, artifacts: dict) -> np.ndarray:
    """
    Mirrors load_and_preprocess_audio exactly, using saved artifact parameters.
    Returns shape (1, N_MELS, T, 1).
    """
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

    y, _     = librosa.load(audio_path, sr=sr, mono=True)
    y_p, _   = librosa.effects.trim(y, top_db=top_db)
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
    plt.close(fig); buf.seek(0)
    return Image.open(buf)


def make_confidence_chart(probs, classes, predicted) -> Image.Image:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#1a1a2e"); ax.set_facecolor("#16213e")
    labels = [f"{EMOTION_EMOJI.get(c, '')} {c}" for c in classes]
    colors = ["#ff6b6b" if c == predicted else "#00d4ff" for c in classes]
    bars   = ax.barh(labels, probs * 100, color=colors, edgecolor="#0f3460", height=0.6)
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
                    f"🤔  **Uncertain** (top: {predicted}, {confidence*100:.1f}%)\n"
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
            # 🎙️ Speech Emotion Recognition  (v3 — optimised)
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
        print("  MODE: TRAIN FROM SCRATCH  (v3)")
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
        print("\n📊 Speaker-independent split:")
        df_train, df_val, df_test = speaker_independent_split(
            df, val_actors=VAL_ACTORS, test_actors=TEST_ACTORS
        )

        # Check all classes present in train
        train_classes = set(df_train["label"].unique())
        if train_classes != set(range(NUM_CLASSES)):
            missing = [CLASS_NAMES[i] for i in sorted(
                set(range(NUM_CLASSES)) - train_classes)]
            print(f"⚠️  Missing from train set: {missing}")
            print("   Consider adjusting VAL_ACTORS / TEST_ACTORS.")

        # 4. Feature extraction (no waveform augmentation — done via tf.data)
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
        print(f"   Train label dist: "
              f"{dict(sorted(Counter(y_train.tolist()).items()))}")
        print(f"   Val   label dist: "
              f"{dict(sorted(Counter(y_val.tolist()).items()))}")

        # 5. Train
        model, history = train_model(X_train, y_train, X_val, y_val, NUM_CLASSES)

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
        artifacts = save_artifacts(model, le, CLASS_NAMES, NUM_CLASSES,
                                   acc, macro_f1)

    else:
        print("\n" + "="*60)
        print("  MODE: LOAD SAVED MODEL")
        print("="*60)
        model, artifacts = load_artifacts()

    # 9. Launch GUI
    launch_gradio(model, artifacts)


if __name__ == "__main__":
    main()
