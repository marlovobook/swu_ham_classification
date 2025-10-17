"""
HAM10000 EfficientNetB0 + Metadata (Thomas et al., 2021 inspired)
- 3-channel RGB input to use ImageNet weights (fixes shape mismatch)
- Stratified 80/10/10 split
- On-the-fly augmentation: rotation, translation, zoom, H/V flips
- Oversampling (replication) on training set via sample_from_datasets
- Metadata branch (age scaled, sex one-hot, localization one-hot)
- EfficientNetB0 image branch, concatenate + MLP head
- Works on VRAM ~16GB with reasonable batch sizes

Adjust the CONFIG block for your paths.
Tested with TensorFlow 2.15+ / Keras 3+, but compatible with 2.10+ with minor tweaks.
"""

import os
import math
import json
import random
import numpy as np
import pandas as pd
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
CONFIG = {
    "IMG_DIRS": [
        r"e:\MSDS\DS533\swu_hum_classification\datasets\HAM10000_images_part_1",
        r"e:\MSDS\DS533\swu_hum_classification\datasets\HAM10000_images_part_2",
    ],
    "METADATA_CSV": r"e:\MSDS\DS533\swu_hum_classification\datasets\HAM10000_metadata.csv",  # columns include: image_id, dx, age, sex, localization
    "IMG_SIZE": 224,
    "BATCH": 32,                 # Use 32–64 if VRAM allows
    "EPOCHS": 50,
    "BACKBONE": "EfficientNetB0", # Keep B0 for VRAM 16GB
    "MIXED_PRECISION": True,     # Set False if you hit numeric issues
    "OUTPUT_DIR": r"e:\MSDS\DS533\swu_hum_classification\outputs_ham10000",
    "SEED": 42,
}

os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

# Silence TF INFO logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Mixed precision (recommended on modern GPUs)
if CONFIG["MIXED_PRECISION"]:
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
    except Exception as e:
        print("[WARN] Mixed precision not available:", e)

# -------------------------------------------------------------
# LABELS
# -------------------------------------------------------------
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

# -------------------------------------------------------------
# LOAD METADATA & BUILD DATAFRAME
# -------------------------------------------------------------

def collect_image_paths(img_dirs):
    paths = []
    for d in img_dirs:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(d, fn))
    return paths

all_paths = collect_image_paths(CONFIG["IMG_DIRS"])
print(f"Found {len(all_paths)} image files from provided folders.")

# Load metadata
md = pd.read_csv(CONFIG["METADATA_CSV"])  # expects columns: image_id, dx(label), age, sex, localization

# map image_id -> full path
# image files names are like ISIC_0024306.jpg ; metadata has image_id without extension
name2path: Dict[str, str] = {}
for p in all_paths:
    base = os.path.basename(p)
    stem = os.path.splitext(base)[0]
    name2path[stem] = p

# Build dataframe by joining metadata with resolved paths
# Filter to our 7 dx classes
md = md[md["dx"].isin(CLASSES)].copy()
md["label_idx"] = md["dx"].map(CLASS2IDX).astype(int)

# Fix pandas chained assignment: assign back safely
if md["age"].isna().all():
    # Rare case, fallback to 45
    md["age"] = 45
else:
    md["age"] = md["age"].fillna(md["age"].median())

# Normalize categorical entries
md["sex"] = md["sex"].fillna("unknown").str.lower()
md["localization"] = md["localization"].fillna("unknown").str.lower()

# Resolve paths; drop rows with missing files
md["image_path"] = md["image_id"].map(name2path)
md = md[~md["image_path"].isna()].copy()

print("Classes:", CLASSES)
print("Total images:", len(md))

# -------------------------------------------------------------
# STRATIFIED SPLIT 80/10/10
# -------------------------------------------------------------
train_df, temp_df = train_test_split(
    md, test_size=0.2, stratify=md["label_idx"], random_state=CONFIG["SEED"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label_idx"], random_state=CONFIG["SEED"]
)
print(f"Split -> train:{len(train_df)}  val:{len(val_df)}  test:{len(test_df)}")

# -------------------------------------------------------------
# METADATA ENCODING (age scaling, one-hots)
# -------------------------------------------------------------
# Fit encoders on *all* data to keep consistent dimensions
sex_values = sorted(md["sex"].unique().tolist())
loc_values = sorted(md["localization"].unique().tolist())
SEX2IDX = {s: i for i, s in enumerate(sex_values)}
LOC2IDX = {l: i for i, l in enumerate(loc_values)}

age_scaler = StandardScaler()
age_scaler.fit(md[["age"]].values.astype(np.float32))

meta_dim = 1 + len(SEX2IDX) + len(LOC2IDX)  # age + sex one-hot + loc one-hot

# Helper to convert one row -> meta vector

def row_to_meta(row) -> np.ndarray:
    age = row["age"]
    sex = row["sex"]
    loc = row["localization"]

    age_scaled = age_scaler.transform([[float(age)]])[0, 0]

    sex_oh = np.zeros(len(SEX2IDX), dtype=np.float32)
    sex_oh[SEX2IDX.get(sex, 0)] = 1.0

    loc_oh = np.zeros(len(LOC2IDX), dtype=np.float32)
    loc_oh[LOC2IDX.get(loc, 0)] = 1.0

    return np.concatenate([[age_scaled], sex_oh, loc_oh]).astype(np.float32)

# Build arrays for tf.data

def build_np_arrays(df):
    paths = df["image_path"].tolist()
    labels = df["label_idx"].astype(np.int32).values
    metas = np.vstack([row_to_meta(r) for _, r in df.iterrows()])
    return np.array(paths), metas, labels

train_paths, train_meta, train_y = build_np_arrays(train_df)
val_paths,   val_meta,   val_y   = build_np_arrays(val_df)
test_paths,  test_meta,  test_y  = build_np_arrays(test_df)

# Save label mapping for later inference
with open(os.path.join(CONFIG["OUTPUT_DIR"], "class_indices.json"), "w", encoding="utf-8") as f:
    json.dump({i: c for c, i in CLASS2IDX.items()}, f, ensure_ascii=False, indent=2)

# -------------------------------------------------------------
# TF.DATA PIPELINES
# -------------------------------------------------------------
IMG_SIZE = CONFIG["IMG_SIZE"]

@tf.function
def preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    # Force decode to 3 channels to match ImageNet weights
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    # EfficientNet expects [0,255] then preprocess_input; if you prefer, use rescaling(1/255.) + normalization below
    return img

# Augment
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),                 # ~±10°
    layers.RandomTranslation(0.1, 0.1),        # 10% shift
    layers.RandomZoom((-0.1, 0.1)),            # zoom in/out
], name="augment")

# EfficientNetB0 preprocess
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
except Exception:
    # Fallback: simple rescale to [0,1]
    effnet_preprocess = lambda x: x / 255.0


def make_dataset(paths, metas, labels, training: bool, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((paths, metas, labels))

    def _map(path, meta, y):
        img = preprocess_image(path)
        if training:
            img = augment(img, training=True)
        # Apply preprocess (expects float32)
        img = effnet_preprocess(img)
        return {"image": img, "meta": meta}, y

    if training:
        ds = ds.shuffle(4096, seed=CONFIG["SEED"], reshuffle_each_iteration=True)
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Per-class datasets for oversampling (replication)

def make_per_class_datasets(paths, metas, labels, batch):
    per_class = []
    paths = np.array(paths)
    metas = np.array(metas)
    labels = np.array(labels)
    for c in range(len(CLASSES)):
        idx = np.where(labels == c)[0]
        ds_c = tf.data.Dataset.from_tensor_slices((paths[idx], metas[idx], labels[idx]))
        ds_c = ds_c.shuffle(len(idx), seed=CONFIG["SEED"], reshuffle_each_iteration=True)
        ds_c = ds_c.repeat()  # allow infinite sampling
        per_class.append(ds_c)
    # Equal weights for each class
    weights = [1.0 / len(CLASSES)] * len(CLASSES)

    def _map(path, meta, y):
        img = preprocess_image(path)
        img = augment(img, training=True)
        img = effnet_preprocess(img)
        return {"image": img, "meta": meta}, y

    balanced = tf.data.Dataset.sample_from_datasets(per_class, weights=weights, seed=CONFIG["SEED"])
    balanced = balanced.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    balanced = balanced.batch(batch).prefetch(tf.data.AUTOTUNE)
    return balanced

# Build datasets
train_ds_balanced = make_per_class_datasets(train_paths, train_meta, train_y, CONFIG["BATCH"])  # Oversampled
val_ds  = make_dataset(val_paths, val_meta, val_y, training=False, batch_size=CONFIG["BATCH"]) 
test_ds = make_dataset(test_paths, test_meta, test_y, training=False, batch_size=CONFIG["BATCH"]) 

# -------------------------------------------------------------
# MODEL
# -------------------------------------------------------------
from tensorflow.keras.applications import EfficientNetB0

# Inputs
img_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
meta_in = layers.Input(shape=(meta_dim,), name="meta")

# Backbone (ImageNet weights) + global pooling
# NOTE: include_top=False and pooling='avg' give a compact feature vector
base = EfficientNetB0(include_top=False, weights=None, input_tensor=img_in, pooling="avg")
img_feat = layers.Dropout(0.2, name="img_dropout")(base.output)

# Metadata branch
m = layers.Dense(64, activation="relu")(meta_in)
m = layers.Dropout(0.2)(m)
m = layers.Dense(32, activation="relu")(m)

# Fuse branches
h = layers.Concatenate(name="fuse")([img_feat, m])
h = layers.Dense(128, activation="relu")(h)
h = layers.Dropout(0.3)(h)

# IMPORTANT: in mixed precision, keep final dtype float32
out = layers.Dense(len(CLASSES), activation="softmax", dtype="float32", name="pred")(h)

model = Model(inputs=[img_in, meta_in], outputs=out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------
steps_per_epoch = math.ceil(len(train_df) / CONFIG["BATCH"])  # with oversampling .repeat(), we need steps

ckpt_path = os.path.join(CONFIG["OUTPUT_DIR"], "best_efficientnetb0_meta.keras")
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True, monitor="val_accuracy"),
    ReduceLROnPlateau(patience=4, factor=0.2, monitor="val_loss", min_lr=1e-6),
    ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True)
]

history = model.fit(
    train_ds_balanced,
    validation_data=val_ds,
    epochs=CONFIG["EPOCHS"],
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks,
    verbose=1
)

# Save training history
hist_path = os.path.join(CONFIG["OUTPUT_DIR"], "train_history.json")
with open(hist_path, "w") as f:
    json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

# -------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------
print("Evaluating on test set …")
results = model.evaluate(test_ds, verbose=1)
print(dict(zip(model.metrics_names, results)))

# Save model
final_path = os.path.join(CONFIG["OUTPUT_DIR"], "final_efficientnetb0_meta.keras")
model.save(final_path)
print("Saved model to:", final_path)

# -------------------------------------------------------------
# NOTES
# -------------------------------------------------------------
# - If you still see oneDNN logs, it's just informational. To disable oneDNN completely:
#   os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  (set BEFORE importing tensorflow)
# - If you prefer class_weight instead of replication oversampling:
#   from sklearn.utils.class_weight import compute_class_weight
#   cw = compute_class_weight('balanced', classes=np.arange(len(CLASSES)), y=train_y)
#   class_weight = {i: float(w) for i, w in enumerate(cw)}
#   model.fit(train_ds, validation_data=val_ds, class_weight=class_weight, ...)
# - To switch to EfficientNetB3, just import EfficientNetB3 and keep 3-channel input. Increase IMG_SIZE (e.g., 300) and reduce BATCH if OOM.
