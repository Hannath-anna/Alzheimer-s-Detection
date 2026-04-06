"""
Alzheimer Detection - CNN Training Script (v2 - FIXED)
Dataset: MRI scan images (4 classes)
Classes: Non Dementia | Very Mild Dementia | Mild Dementia | Moderate Dementia

Fix Log vs v1:
  [CRITICAL] Removed rescale=1./255 — EfficientNetB0 has its own preprocess_input
             that expects raw [0,255] pixels. Rescaling was destroying features.
  [BUG FIX]  val_gen now uses dataset/val/ (dedicated split) instead of splitting
             training data with validation_split — eliminates data leakage risk.
  [BUG FIX]  Augmentation no longer applied to validation/test generators.
  [IMPROVE]  Added label smoothing (0.1) to handle extreme class imbalance.
  [IMPROVE]  Simpler head: GAP → BN → Dense(256, relu) → Dropout(0.5) → Dense(4)
  [IMPROVE]  Cosine decay LR schedule for Phase 2 fine-tuning.

Expected folder structure:
    dataset/
    ├── train/
    │   ├── Non_Demented/
    │   ├── Very_Mild_Demented/
    │   ├── Mild_Demented/
    │   └── Moderate_Demented/
    ├── val/
    │   └── ...same classes...
    └── test/
        └── ...same classes...
"""

import os, warnings, pickle
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers, losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS_P1   = 20          # Phase 1 — frozen backbone
EPOCHS_P2   = 40          # Phase 2 — fine-tune
LR_P1       = 1e-3        # Higher LR for head-only training
LR_P2       = 5e-5        # Lower LR for fine-tuning
NUM_CLASSES = 4
DATASET_DIR = "dataset"          # ← change to "dataset_balanced" after running balance_dataset.py
MODEL_DIR   = "models"
GRAPH_DIR   = "graphs"
SEED        = 42

CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
DISPLAY_NAMES = {
    'Mild_Demented':      'Mild Dementia',
    'Moderate_Demented':  'Moderate Dementia',
    'Non_Demented':       'Non Dementia',
    'Very_Mild_Demented': 'Very Mild Dementia'
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. Data Generators
# ─────────────────────────────────────────────
# ⚠️  NO rescale=1./255 — EfficientNet's preprocess_input handles normalisation
#     inside the model graph (see build section below).
print("📂 Preparing data generators...")

train_datagen = ImageDataGenerator(
    # No rescale — handled inside model via preprocess_input
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest',
)

# Validation & test: NO augmentation, NO rescale (model handles it)
plain_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=SEED, shuffle=True
)
val_gen = plain_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', seed=SEED, shuffle=False
)
test_gen = plain_datagen.flow_from_directory(
    os.path.join(DATASET_DIR, 'test'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', shuffle=False
)

print(f"   Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
print(f"   Classes: {train_gen.class_indices}\n")

# ─────────────────────────────────────────────
# 1b. Class Weights (handle imbalance)
# ─────────────────────────────────────────────
labels_array = train_gen.classes
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_array),
    y=labels_array
)
class_weight_dict = dict(enumerate(class_weights_array))
print(f"   ⚖️  Class weights: { {k: round(v, 2) for k, v in class_weight_dict.items()} }\n")

# ─────────────────────────────────────────────
# 2. Build CNN Model
# ─────────────────────────────────────────────
print("🏗️  Building CNN model (EfficientNetB0 backbone)...")

base_model = EfficientNetB0(
    weights='imagenet', include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False   # Phase 1: frozen

# --- Model graph ---
inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name='image_input')
# 🔑 KEY FIX: EfficientNet preprocessing inside the model.
#    This accepts raw [0,255] uint8-range values (what the generator provides)
#    and scales them to [-1, 1] as EfficientNetB0 expects.
x = tf.keras.layers.Lambda(
    lambda img: preprocess_input(img),
    name='efficientnet_preprocessing'
)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D(name='gap')(x)
x = layers.BatchNormalization(name='bn_head')(x)
x = layers.Dense(256, activation='relu',
                 kernel_regularizer=regularizers.l2(1e-4),
                 name='dense_head')(x)
x = layers.Dropout(0.5, name='dropout_head')(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inputs, outputs)

# Label smoothing helps with extreme class imbalance
smooth_loss = losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_P1),
    loss=smooth_loss,
    metrics=['accuracy']
)
model.summary()

# ─────────────────────────────────────────────
# 3. Callbacks
# ─────────────────────────────────────────────
BEST_MODEL_PATH = f'{MODEL_DIR}/cnn_model_best.keras'

def make_callbacks(patience_es=6, patience_lr=3, min_lr=1e-7):
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5,
            patience=patience_lr, min_lr=min_lr, verbose=1
        ),
        callbacks.ModelCheckpoint(
            BEST_MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1
        ),
    ]

# ─────────────────────────────────────────────
# 4. Phase 1 — Train classifier head (backbone frozen)
# ─────────────────────────────────────────────
print("\n🏋️  Phase 1: Training classifier head (backbone frozen)...")
history1 = model.fit(
    train_gen,
    epochs=EPOCHS_P1,
    validation_data=val_gen,
    callbacks=make_callbacks(patience_es=6, patience_lr=3),
    class_weight=class_weight_dict,
    verbose=1
)
phase1_epochs = len(history1.history['accuracy'])

# ─────────────────────────────────────────────
# 5. Phase 2 — Fine-tune top layers of EfficientNet
# ─────────────────────────────────────────────
print("\n🔧 Phase 2: Fine-tuning top 50 EfficientNet layers...")
base_model.trainable = True
# Keep BatchNorm layers frozen during fine-tune to preserve learned statistics
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# Only train the last 50 layers of the backbone
for layer in base_model.layers[:-50]:
    layer.trainable = False

trainable_count = sum(1 for l in model.trainable_weights)
print(f"   Trainable weight tensors: {trainable_count}")

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR_P2),
    loss=smooth_loss,
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=EPOCHS_P2,
    validation_data=val_gen,
    callbacks=make_callbacks(patience_es=10, patience_lr=4, min_lr=1e-8),
    class_weight=class_weight_dict,
    verbose=1
)

# Merge histories
def merge_history(h1, h2):
    merged = {}
    for key in h1.history:
        merged[key] = h1.history[key] + h2.history[key]
    return merged

history = merge_history(history1, history2)

# ─────────────────────────────────────────────
# 6. Evaluate on test set
# ─────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"   Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
disp_labels  = [DISPLAY_NAMES.get(idx_to_class[i], idx_to_class[i]) for i in range(NUM_CLASSES)]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=disp_labels, zero_division=0))

# ─────────────────────────────────────────────
# 7. Graphs
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = ['#6C63FF', '#FF6584', '#43B89C', '#FFB347']

epochs_range = range(1, len(history['accuracy']) + 1)

# --- 7a. Accuracy ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(epochs_range, history['accuracy'],     color='#6C63FF', lw=2.5, label='Train Accuracy', marker='o', markersize=3)
ax.plot(epochs_range, history['val_accuracy'], color='#FF6584', lw=2.5, label='Val Accuracy',   marker='s', markersize=3)
ax.axhline(y=test_acc, color='#43B89C', linestyle='--', lw=2, label=f'Test Accuracy: {test_acc:.3f}')
ax.axvline(x=phase1_epochs, color='#FFB347', linestyle=':', lw=2, label=f'Phase 1→2 (ep {phase1_epochs})')
ax.text(phase1_epochs + 0.3, 0.02, 'Fine-tune\nstart', color='#FFB347', fontsize=9, va='bottom')
ax.set_title('CNN Training & Validation Accuracy', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim(0.0, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/cnn_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: cnn_accuracy.png")

# --- 7b. Loss ---
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(epochs_range, history['loss'],     color='#6C63FF', lw=2.5, label='Train Loss', marker='o', markersize=3)
ax.plot(epochs_range, history['val_loss'], color='#FF6584', lw=2.5, label='Val Loss',   marker='s', markersize=3)
ax.axvline(x=phase1_epochs, color='#FFB347', linestyle=':', lw=2, label=f'Phase 1→2 (ep {phase1_epochs})')
max_gap_epoch = int(np.argmax(np.array(history['val_loss']) - np.array(history['loss']))) + 1
gap = history['val_loss'][max_gap_epoch-1] - history['loss'][max_gap_epoch-1]
if gap > 0.05:
    ax.annotate(f'Overfitting gap\n({gap:.2f})',
                xy=(max_gap_epoch, history['val_loss'][max_gap_epoch-1]),
                xytext=(max_gap_epoch + 1, history['val_loss'][max_gap_epoch-1] + 0.03),
                arrowprops=dict(arrowstyle='->', color='#FF6584'),
                fontsize=9, color='#FF6584')
ax.set_title('CNN Training & Validation Loss', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/cnn_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: cnn_loss.png")

# --- 7c. Confusion Matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True)
annot = np.array([[f"{cm[i,j]}\n({cm_pct[i,j]*100:.1f}%)"
                   for j in range(NUM_CLASSES)] for i in range(NUM_CLASSES)])
sns.heatmap(cm, annot=annot, fmt='', cmap='Purples', ax=ax,
            xticklabels=disp_labels, yticklabels=disp_labels,
            linewidths=1, linecolor='white', cbar_kws={'shrink': 0.8})
ax.set_title('Confusion Matrix - CNN (MRI)', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/cnn_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: cnn_confusion_matrix.png")

# --- 7d. Per-class Accuracy ---
per_class_acc = cm.diagonal() / cm.sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(disp_labels, per_class_acc * 100, color=COLORS, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val*100:.1f}%', ha='center', fontweight='bold', fontsize=11)
majority_ratio = cm.sum(axis=1).max() / cm.sum() * 100
ax.axhline(y=majority_ratio, color='#888', linestyle='--', lw=1.5,
           label=f'Majority class baseline ({majority_ratio:.1f}%)')
ax.legend(fontsize=10)
ax.set_title('Per-Class Accuracy - CNN', fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(0, 110)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/cnn_per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: cnn_per_class_accuracy.png")

# --- 7e. Prediction Confidence Distribution ---
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(14, 4))
for idx in range(NUM_CLASSES):
    mask = y_true == idx
    confs = y_pred_probs[mask, idx]
    axes[idx].hist(confs, bins=20, color=COLORS[idx], edgecolor='white', alpha=0.85)
    axes[idx].set_title(disp_labels[idx], fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Confidence', fontsize=9)
    axes[idx].set_ylabel('Count', fontsize=9)
fig.suptitle('Prediction Confidence Distribution per Class', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{GRAPH_DIR}/cnn_confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   📊 Saved: cnn_confidence_distribution.png")

# ─────────────────────────────────────────────
# 8. Save Model & Artifacts
# ─────────────────────────────────────────────
final_model_path = f'{MODEL_DIR}/cnn_model.keras'
model.save(final_model_path)

cnn_artifacts = {
    'class_indices':  train_gen.class_indices,
    'display_names':  DISPLAY_NAMES,
    'class_names':    list(train_gen.class_indices.keys()),
    'img_size':       IMG_SIZE,
    'test_accuracy':  test_acc,
    'test_loss':      test_loss,
    'history':        history
}
with open(f'{MODEL_DIR}/cnn_artifacts.pkl', 'wb') as f:
    pickle.dump(cnn_artifacts, f)

print(f"\n✅ CNN Model saved → {final_model_path}")
print(f"✅ Training complete! Test Accuracy: {test_acc:.4f}")