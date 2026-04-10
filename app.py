"""
AlzheimerAI — Flask Web Application  v2.0
Features:
  · ML Clinical Analysis  → 3-model Ensemble Voting (RF + GB + SVM)
  · SHAP Explainability   → Per-patient waterfall chart (manual TreeSHAP)
  · MRI CNN Analysis      → EfficientNetB0 + Grad-CAM heatmap overlay
"""
import os, pickle, warnings, glob, time, io, base64
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "alzheimer_ai_v2_2024"
app.config['UPLOAD_FOLDER']      = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# ══════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════
ML_MODELS    = {}
ML_SCALER    = None
ML_ARTIFACTS = {}
CNN_MODEL    = None
CNN_ARTIFACTS = {}

def load_ml_models():
    global ML_MODELS, ML_SCALER, ML_ARTIFACTS
    try:
        ML_SCALER = joblib.load('models/ml_scaler.pkl')
        with open('models/ml_artifacts.pkl', 'rb') as f:
            ML_ARTIFACTS = pickle.load(f)
        ML_ARTIFACTS['test_accuracy'] = float(ML_ARTIFACTS.get('test_accuracy', 0))
        ML_ARTIFACTS['roc_auc']       = float(ML_ARTIFACTS.get('roc_auc', 0))
        paths = {
            'Random Forest':     'models/ensemble_random_forest.pkl',
            'Gradient Boosting': 'models/ensemble_gradient_boosting.pkl',
            'SVM':               'models/ensemble_svm.pkl',
        }
        for name, path in paths.items():
            if os.path.exists(path):
                ML_MODELS[name] = joblib.load(path)
        print(f"✅ Ensemble ML loaded: {list(ML_MODELS.keys())}")
    except Exception as e:
        print(f"⚠️  ML load error: {e}")

def load_cnn_artifacts():
    global CNN_ARTIFACTS
    try:
        with open('models/cnn_artifacts.pkl', 'rb') as f:
            CNN_ARTIFACTS = pickle.load(f)
        acc = CNN_ARTIFACTS.get('test_accuracy', 0)
        CNN_ARTIFACTS['test_accuracy'] = float(acc) if float(acc) <= 1.0 else float(acc)/100.0
        print(f"✅ CNN artifacts loaded  acc={CNN_ARTIFACTS['test_accuracy']*100:.2f}%")
    except Exception as e:
        print(f"⚠️  CNN artifacts: {e}")

def load_cnn_model():
    global CNN_MODEL
    if not os.path.exists('models/cnn_model.keras'):
        print("⚠️  cnn_model.keras not found"); return
    try:
        import tensorflow as tf
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.applications.efficientnet import preprocess_input
        from tensorflow.keras import layers, regularizers

        # The saved model contains a Lambda layer that Keras 3 cannot deserialize
        # because it cannot automatically infer the Lambda's output shape.
        #
        # FIX: Rebuild the *exact* same architecture (matching train_cnn.py) using a
        #      proper Layer subclass instead of Lambda, then load weights by position.
        #      This avoids the deserialization problem entirely.

        @tf.keras.utils.register_keras_serializable(package='alzheimer')
        class EfficientNetPreprocessing(tf.keras.layers.Layer):
            """Drop-in replacement for the Lambda(preprocess_input) layer."""
            def call(self, inputs):
                return preprocess_input(inputs)
            def compute_output_shape(self, input_shape):
                return input_shape

        IMG_SIZE    = CNN_ARTIFACTS.get('img_size', (224, 224))
        NUM_CLASSES = len(CNN_ARTIFACTS.get('class_indices', {1:0, 2:0, 3:0, 4:0}))
        if NUM_CLASSES == 0:
            NUM_CLASSES = 4

        base_model = EfficientNetB0(
            weights='imagenet', include_top=False,
            input_shape=(*IMG_SIZE, 3)
        )
        inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3), name='image_input')
        x       = EfficientNetPreprocessing(name='efficientnet_preprocessing')(inputs)
        x       = base_model(x, training=False)
        x       = layers.GlobalAveragePooling2D(name='gap')(x)
        x       = layers.BatchNormalization(name='bn_head')(x)
        x       = layers.Dense(256, activation='relu',
                               kernel_regularizer=regularizers.l2(1e-4),
                               name='dense_head')(x)
        x       = layers.Dropout(0.5, name='dropout_head')(x)
        outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)
        model   = tf.keras.Model(inputs, outputs)

        # Load weights from the saved file (skip the Lambda/object config entirely)
        model.load_weights('models/cnn_model.keras')
        CNN_MODEL = model
        print("✅ CNN model loaded (architecture rebuilt + weights restored)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️  CNN model: {e}")

load_ml_models()
load_cnn_artifacts()
load_cnn_model()

# ══════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════
def allowed_image(fn):
    return '.' in fn and fn.rsplit('.', 1)[1].lower() in ALLOWED_IMG

def risk_meta(prob):
    if prob < 0.30: return 'Low Risk',      'none',   '#22c55e'
    if prob < 0.60: return 'Moderate Risk', 'medium', '#f59e0b'
    return                  'High Risk',    'high',   '#ef4444'

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64

def cleanup_old_uploads(max_age=3600):
    now = time.time()
    for f in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
        if os.path.isfile(f) and (now - os.path.getmtime(f)) > max_age:
            try: os.remove(f)
            except: pass

# ══════════════════════════════════════════════
#  SHAP  (manual TreeSHAP via decision path tracing)
# ══════════════════════════════════════════════
from sklearn.tree import _tree

def _tree_contributions(tree, X_f32, n_features):
    node_indicator = tree.decision_path(X_f32)
    node_ids       = node_indicator.indices
    feature        = tree.feature
    contribs       = np.zeros(n_features)
    for i in range(len(node_ids) - 1):
        nid      = node_ids[i]
        feat_idx = feature[nid]
        if feat_idx == _tree.TREE_UNDEFINED:
            continue
        n_node   = tree.n_node_samples[nid]
        child_id = node_ids[i + 1]
        n_child  = tree.n_node_samples[child_id]
        p_parent = tree.value[nid][0][1] / n_node
        p_child  = tree.value[child_id][0][1] / n_child
        contribs[feat_idx] += p_child - p_parent
    return contribs

def compute_shap(rf_model, X_scaled_row, feature_names):
    X_f32     = X_scaled_row.astype(np.float32).reshape(1, -1)
    n         = len(feature_names)
    all_c     = np.array([_tree_contributions(e.tree_, X_f32, n) for e in rf_model.estimators_])
    shap_vals = np.mean(all_c, axis=0)
    base_val  = np.mean([
        e.tree_.value[0][0][1] / e.tree_.n_node_samples[0]
        for e in rf_model.estimators_
    ])
    return shap_vals, base_val

def make_shap_waterfall(shap_vals, base_value, feature_names, feature_vals,
                        top_n=12, final_prob=None):
    idx  = np.argsort(np.abs(shap_vals))[::-1][:top_n]
    vals = shap_vals[idx]
    nms  = [feature_names[i] for i in idx]
    raw  = [feature_vals[i]  for i in idx]
    order = np.argsort(vals)
    vals  = vals[order];  nms = [nms[i] for i in order];  raw = [raw[i] for i in order]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    colors = ['#ef4444' if v > 0 else '#22c55e' for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.62, edgecolor='none', alpha=0.90)

    for i, (v, r) in enumerate(zip(vals, raw)):
        sign = '+' if v > 0 else ''
        ax.text(v + (0.0005 if v >= 0 else -0.0005), i,
                f' {sign}{v:.4f}   val={r:.2f}',
                va='center', ha='left' if v >= 0 else 'right',
                color='#c9d1d9', fontsize=8.5, fontfamily='monospace')

    ax.set_yticks(range(len(nms)))
    ax.set_yticklabels(nms, color='#c9d1d9', fontsize=10)
    ax.set_xlabel('SHAP Value  (positive → higher Alzheimer risk)', color='#8b949e', fontsize=10)
    ax.axvline(0, color='#30363d', linewidth=1.8)
    ax.tick_params(axis='x', colors='#8b949e')
    ax.spines[:].set_visible(False)
    ax.grid(axis='x', color='#161b22', linewidth=0.8, linestyle='--', alpha=0.8)
    red_p   = mpatches.Patch(color='#ef4444', alpha=0.9, label='↑ Pushes toward Alzheimer')
    green_p = mpatches.Patch(color='#22c55e', alpha=0.9, label='↓ Pushes toward No Alzheimer')
    ax.legend(handles=[red_p, green_p], loc='lower right',
              framealpha=0.08, labelcolor='#c9d1d9', fontsize=9,
              facecolor='#161b22', edgecolor='#30363d')
    prob_str = f'  ·  Final P(Alzheimer) = {final_prob:.1%}' if final_prob is not None else ''
    ax.set_title(f'SHAP Waterfall Chart — Patient-Level Explanation{prob_str}\n'
                 f'Base value = {base_value:.4f}  |  showing top {top_n} features',
                 color='#e6edf3', fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()
    return fig_to_b64(fig)

# ══════════════════════════════════════════════
#  GRAD-CAM  (Keras 3 compatible)
#
#  WHY the old approach broke:
#    tf.keras.Model(inputs=model.inputs,
#                   outputs=[sub_model.get_layer('top_conv').output, model.output])
#  fails in Keras 3 because sub_model.get_layer('top_conv').output is a tensor
#  in the SUB-MODEL's internal graph — not reachable from the top-level model.inputs.
#  Keras 3 detects this mismatch and raises Functional.call() with a tensor-id error.
#
#  FIX: Never build a new Model that crosses graph boundaries.
#  Instead:
#    Part A  –  sub_model.input → top_conv.output   (valid: same sub-model graph)
#    Part B  –  call subsequent layers ONE BY ONE through GradientTape so that
#               loss is computed through the watched conv_output tensor.
# ══════════════════════════════════════════════

# EfficientNetB0 preferred layer names for Grad-CAM (most → least preferred)
_GRADCAM_CANDIDATES = [
    'top_conv',
    'block7a_project_conv',
    'block6d_project_conv',
    'block7a_expand_conv',
    'block6d_expand_conv',
]

def _collect_all_layers(model):
    """Recursively collect (layer, parent_model) for every layer including nested ones."""
    pairs = []
    for layer in model.layers:
        pairs.append((layer, model))
        if hasattr(layer, 'layers'):
            pairs.extend(_collect_all_layers(layer))
    return pairs

def _find_submodel_and_layer(model):
    """
    Return (sub_model, layer_name, conv_layer_idx_in_submodel).
    sub_model is the nested Functional model (EfficientNetB0).
    conv_layer_idx is the position of target layer inside sub_model.layers.
    """
    all_pairs = _collect_all_layers(model)

    # Try preferred candidates first
    for candidate in _GRADCAM_CANDIDATES:
        for layer, parent in all_pairs:
            if layer.name == candidate and hasattr(parent, 'layers') and parent is not model:
                idx = [i for i, l in enumerate(parent.layers) if l.name == candidate]
                if idx:
                    print(f"   Grad-CAM: found '{candidate}' in sub-model '{parent.name}'")
                    return parent, candidate, idx[0]

    # Generic fallback: deepest conv with 4-D spatial output inside any sub-model
    for layer, parent in reversed(all_pairs):
        if parent is model:
            continue
        if 'conv' not in layer.name.lower():
            continue
        try:
            shape = layer.output_shape
            if len(shape) == 4 and shape[1] and shape[1] > 1:
                idx = [i for i, l in enumerate(parent.layers) if l.name == layer.name]
                if idx:
                    print(f"   Grad-CAM: fallback layer '{layer.name}' in '{parent.name}'")
                    return parent, layer.name, idx[0]
        except Exception:
            pass

    return None, None, None

def compute_gradcam(model, img_array, class_idx, *_ignored):
    """
    Keras 3-compatible Grad-CAM via graph-split + manual layer replay.

    Algorithm
    ---------
    1. Locate the EfficientNet sub-model and the target conv layer inside it.
    2. Build  part_A = Model(sub.input → conv.output)   [single sub-model graph — valid]
    3. Collect all layers that come AFTER the conv layer:
         • remaining layers in the sub-model  (top_bn, top_activation, …)
         • then every non-input layer in the top-level model that follows the sub-model
           (GlobalAveragePooling, Dropout, Dense, …)
    4. Run GradientTape:
         conv_out = part_A(img)        — intermediate activation
         tape.watch(conv_out)
         x = conv_out
         x = layer1(x); x = layer2(x); …   — replays the rest of the network
         loss = x[:, class_idx]
       Because loss flows through conv_out, tape.gradient(loss, conv_out) is valid.
    5. Compute standard Grad-CAM:  weights = mean(grads, spatial) ; cam = ReLU(Σ w·A)
    """
    import tensorflow as tf

    img_t = tf.cast(img_array, tf.float32)

    sub_model, conv_name, conv_idx = _find_submodel_and_layer(model)

    if sub_model is None:
        raise RuntimeError(
            "Grad-CAM: could not find a suitable conv layer inside any sub-model. "
            "Visit /debug/cnn to inspect the model architecture."
        )

    # ── Part A: sub_model.input → conv_layer output ─────────────
    part_a = tf.keras.Model(
        inputs  = sub_model.input,
        outputs = sub_model.get_layer(conv_name).output,
        name    = 'gradcam_part_a',
    )

    # ── Collect Part B layers ────────────────────────────────────
    # (a) Remaining layers inside the sub-model after the conv layer
    sub_layers_after = [
        l for l in sub_model.layers[conv_idx + 1:]
        if not isinstance(l, tf.keras.layers.InputLayer)
    ]

    # (b) Top-level layers AFTER the sub-model (classifier head)
    reached_sub = False
    top_tail_layers = []
    for l in model.layers:
        if isinstance(l, tf.keras.layers.InputLayer):
            continue
        if l.name == sub_model.name:
            reached_sub = True
            continue
        if reached_sub:
            top_tail_layers.append(l)

    all_replay_layers = sub_layers_after + top_tail_layers
    print(f"   Grad-CAM: replaying {len(sub_layers_after)} sub-model layers "
          f"+ {len(top_tail_layers)} head layers")

    # ── GradientTape forward pass ────────────────────────────────
    with tf.GradientTape() as tape:
        conv_out = part_a(img_t, training=False)   # (1, H, W, C)
        tape.watch(conv_out)

        x = conv_out
        for layer in all_replay_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)                        # some layers ignore training=

        loss = x[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError(
            "Grad-CAM: tape.gradient returned None. "
            "The replay path may include a non-differentiable op or a layer with multiple inputs. "
            "Visit /debug/cnn for the full layer list."
        )

    # ── Standard Grad-CAM aggregation ───────────────────────────
    weights  = tf.reduce_mean(grads[0], axis=(0, 1)).numpy()   # (C,)
    conv_np  = conv_out[0].numpy()                              # (H, W, C)
    cam      = np.sum(conv_np * weights, axis=-1)               # (H, W)
    cam      = np.maximum(cam, 0)
    cam      = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def overlay_gradcam(original_path, cam, alpha=0.45):
    """Render side-by-side: original | heatmap | overlay. Returns base64 PNG."""
    import matplotlib.cm as mpl_cm

    orig = Image.open(original_path).convert('RGB')
    w, h = orig.size

    # Resize CAM to original image dimensions
    cam_uint8   = (cam * 255).astype(np.uint8)
    cam_resized = np.array(
        Image.fromarray(cam_uint8).resize((w, h), resample=Image.BILINEAR)
    ) / 255.0

    # Apply jet colormap
    heat_rgb = (mpl_cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_rgb)
    blended  = Image.blend(orig, heat_img, alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0d1117')

    panels = [
        ('Original MRI Scan',   np.array(orig)),
        ('Grad-CAM Activation', heat_rgb),
        ('Heatmap Overlay',     np.array(blended)),
    ]
    for ax, (title, im) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, color='#e6edf3', fontsize=11, fontweight='bold', pad=10)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Colorbar only on the heatmap panel
    sm   = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.042, pad=0.03)
    cbar.set_label('Activation Intensity', color='#8b949e', fontsize=8.5)
    cbar.ax.yaxis.set_tick_params(color='#8b949e', length=0)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=7.5)

    # Annotation explaining heatmap colours
    fig.text(0.5, -0.03,
             '🔴 Red = high activation (key decision region)   '
             '🟡 Yellow = medium   🔵 Blue = low activation',
             ha='center', color='#8b949e', fontsize=9)

    fig.suptitle('Grad-CAM: Brain Regions Driving the Prediction',
                 color='#e6edf3', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig_to_b64(fig)

# ══════════════════════════════════════════════
#  CNN MAPS
# ══════════════════════════════════════════════
CNN_DISPLAY = {
    'Mild_Demented':     'Mild Dementia',
    'Moderate_Demented': 'Moderate Dementia',
    'Non_Demented':      'Non Dementia',
    'Very_Mild_Demented':'Very Mild Dementia',
}
CNN_SEVERITY = {
    'Non_Demented':      ('No Dementia Detected', 'none'),
    'Very_Mild_Demented':('Very Mild Dementia',   'low'),
    'Mild_Demented':     ('Mild Dementia',         'medium'),
    'Moderate_Demented': ('Moderate Dementia',     'high'),
}

# ══════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html',
        ml_ready  = len(ML_MODELS) > 0,
        cnn_ready = CNN_MODEL is not None,
        ml_acc    = round(ML_ARTIFACTS.get('test_accuracy', 0) * 100, 1),
        cnn_acc   = round(CNN_ARTIFACTS.get('test_accuracy', 0) * 100, 1) if CNN_ARTIFACTS else None,
    )

@app.route('/ml')
def ml_form():
    if not ML_MODELS:
        flash('ML models not found. Run train_ml.py first.', 'danger')
        return redirect(url_for('index'))
    return render_template('ml_predict.html',
                           features=ML_ARTIFACTS.get('feature_names', []))

@app.route('/ml/predict', methods=['POST'])
def ml_predict():
    if not ML_MODELS:
        flash('ML models not available.', 'danger')
        return redirect(url_for('index'))
    try:
        features     = ML_ARTIFACTS['feature_names']
        feature_vals = [float(request.form.get(f, 0)) for f in features]
        X            = np.array(feature_vals).reshape(1, -1)
        X_scaled     = ML_SCALER.transform(X)

        model_results = []
        votes         = []
        for name, model in ML_MODELS.items():
            proba    = model.predict_proba(X_scaled)[0]
            pred     = int(np.argmax(proba))
            votes.append(pred)
            ens_info = ML_ARTIFACTS.get('ensemble_results', {}).get(name, {})
            prob_pos = round(float(proba[1]) * 100, 2)
            prob_neg = round(100 - prob_pos, 2)   # guarantee sum = 100, not independent rounding
            model_results.append({
                'name':       name,
                'prediction': pred,
                'label':      'Alzheimer' if pred == 1 else 'No Alzheimer',
                'prob_pos':   prob_pos,
                'prob_neg':   prob_neg,
                'confidence': round(float(max(proba)) * 100, 2),
                'test_acc':   round(float(ens_info.get('acc', 0)) * 100, 1),
                'auc':        round(float(ens_info.get('auc', 0)) * 100, 1),
            })

        final_vote   = int(np.bincount(votes).argmax())
        vote_count   = votes.count(final_vote)
        final_label  = 'Alzheimer Detected' if final_vote == 1 else 'No Alzheimer Detected'
        avg_prob_pos = round(float(np.mean([r['prob_pos'] for r in model_results])), 2)
        avg_prob_neg = round(100 - avg_prob_pos, 2)
        risk, risk_lvl, risk_color = risk_meta(avg_prob_pos / 100)
        # Banner colour tracks actual probability risk level, not just binary vote
        final_badge  = risk_lvl

        shap_chart = None
        rf = ML_MODELS.get('Random Forest')
        if rf:
            shap_vals, base_val = compute_shap(rf, X_scaled[0], features)
            shap_chart = make_shap_waterfall(
                shap_vals, base_val, features, feature_vals,
                top_n=12, final_prob=avg_prob_pos / 100
            )

        return render_template('ml_result.html',
            final_label=final_label, final_badge=final_badge,
            final_vote=final_vote, vote_count=vote_count,
            total_models=len(ML_MODELS),
            avg_prob_pos=round(avg_prob_pos, 2),
            avg_prob_neg=round(avg_prob_neg, 2),
            risk=risk, risk_lvl=risk_lvl, risk_color=risk_color,
            model_results=model_results, shap_chart=shap_chart,
            ml_acc=round(ML_ARTIFACTS.get('test_accuracy', 0) * 100, 1),
            ml_auc=round(ML_ARTIFACTS.get('roc_auc', 0) * 100, 1),
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f'Prediction error: {str(e)}', 'danger')
        return redirect(url_for('ml_form'))

@app.route('/cnn')
def cnn_form():
    return render_template('cnn_predict.html', cnn_ready=CNN_MODEL is not None)

@app.route('/cnn/predict', methods=['POST'])
def cnn_predict():
    if CNN_MODEL is None:
        flash('CNN model (cnn_model.keras) not found. Place it in models/ and restart.', 'danger')
        return redirect(url_for('cnn_form'))
    if not CNN_ARTIFACTS.get('class_indices'):
        flash('CNN artifacts missing.', 'danger')
        return redirect(url_for('cnn_form'))
    if 'mri_image' not in request.files or request.files['mri_image'].filename == '':
        flash('No image selected.', 'warning')
        return redirect(url_for('cnn_form'))

    file = request.files['mri_image']
    if not allowed_image(file.filename):
        flash('Invalid file type. Use PNG, JPG, JPEG, BMP or TIFF.', 'danger')
        return redirect(url_for('cnn_form'))

    try:
        base, ext = os.path.splitext(secure_filename(file.filename))
        filename  = f"{base}_{int(time.time())}{ext}"
        filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        cleanup_old_uploads()

        try:
            with Image.open(filepath) as t: t.verify()
        except Exception as e:
            raise ValueError(f"Cannot read image: {e}")

        img_size = CNN_ARTIFACTS.get('img_size', (224, 224))
        img      = Image.open(filepath).convert('RGB').resize(img_size)
        img_arr  = np.expand_dims(np.array(img, dtype=np.float32), 0)   # raw [0,255] — model does preprocess_input internally

        preds         = CNN_MODEL.predict(img_arr, verbose=0)[0]
        class_indices = CNN_ARTIFACTS['class_indices']
        idx_to_class  = {v: k for k, v in class_indices.items()}
        pred_idx      = int(np.argmax(preds))
        pred_class    = idx_to_class.get(pred_idx, 'Unknown')
        confidence    = float(preds[pred_idx])
        display_name  = CNN_DISPLAY.get(pred_class, pred_class)
        severity, badge = CNN_SEVERITY.get(pred_class, ('Unknown', 'medium'))

        # Semantic color fixed per class — never changes based on sort order
        _class_color = {
            'Non_Demented':       ('var(--green)', 'prog-green'),
            'Very_Mild_Demented': ('var(--cyan)',  'prog-cyan'),
            'Mild_Demented':      ('var(--amber)', 'prog-yellow'),
            'Moderate_Demented':  ('var(--red)',   'prog-red'),
        }
        all_probs = sorted([
            {'class':     CNN_DISPLAY.get(idx_to_class.get(i,''), f'Class {i}'),
             'prob':      round(float(preds[i]) * 100, 2),
             'raw_key':   idx_to_class.get(i, f'class_{i}'),
             'txt_color': _class_color.get(idx_to_class.get(i,''), ('var(--text)', 'prog-cyan'))[0],
             'bar_color': _class_color.get(idx_to_class.get(i,''), ('var(--text)', 'prog-cyan'))[1]}
            for i in range(len(preds))
        ], key=lambda x: x['prob'], reverse=True)

        gradcam_b64   = None
        gradcam_error = None
        try:
            # compute_gradcam now handles layer discovery internally
            cam         = compute_gradcam(CNN_MODEL, img_arr, pred_idx)
            gradcam_b64 = overlay_gradcam(filepath, cam, alpha=0.45)
            print("   Grad-CAM: ✅ generated successfully")
        except Exception as ge:
            gradcam_error = str(ge)
            print(f"   Grad-CAM: ❌ {ge}")
            import traceback; traceback.print_exc()

        img_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('cnn_result.html',
            pred_class=display_name, severity=severity, badge=badge,
            confidence=round(confidence * 100, 2),
            all_probs=all_probs, img_url=img_url,
            gradcam_b64=gradcam_b64,
            gradcam_error=gradcam_error,
            cnn_acc=round(CNN_ARTIFACTS.get('test_accuracy', 0) * 100, 1),
        )
    except ValueError as e:
        flash(f'Image error: {str(e)}', 'danger')
        return redirect(url_for('cnn_form'))
    except Exception as e:
        import traceback; traceback.print_exc()
        flash(f'Prediction error: {str(e)}', 'danger')
        return redirect(url_for('cnn_form'))

@app.route('/dashboard')
def dashboard():
    gdir = os.path.join('static', 'graphs')
    ml_map = {
        'ml_class_distribution.png':  ('Class Distribution',        'Alzheimer vs Non-Alzheimer split'),
        'ml_model_comparison.png':    ('Model Accuracy Comparison', '5-fold CV across all ensemble models'),
        'ml_cv_fold_scores.png':      ('CV Fold Scores',            'Per-fold accuracy line chart'),
        'ml_confusion_matrix.png':    ('Confusion Matrix',          'True vs predicted heatmap'),
        'ml_roc_curve.png':           ('ROC Curve',                 f'AUC = {round(ML_ARTIFACTS.get("roc_auc",0)*100,1)}%'),
        'ml_pr_curve.png':            ('Precision-Recall Curve',    'Precision vs Recall tradeoff'),
        'ml_feature_importance.png':  ('Feature Importance',        'Top 20 predictive clinical features'),
        'ml_correlation_heatmap.png': ('Correlation Heatmap',       'Inter-feature correlations'),
    }
    cnn_map = {
        'cnn_accuracy.png':                ('Training Accuracy',       'Train vs validation accuracy'),
        'cnn_loss.png':                    ('Training Loss',           'Train vs validation loss'),
        'cnn_confusion_matrix.png':        ('Confusion Matrix',        '4-class MRI prediction'),
        'cnn_per_class_accuracy.png':      ('Per-Class Accuracy',      'Accuracy per dementia stage'),
        'cnn_confidence_distribution.png': ('Confidence Distribution', 'Confidence histograms per class'),
    }
    ml_graphs  = [{'file': f, 'title': t, 'desc': d} for f,(t,d) in ml_map.items()
                   if os.path.exists(os.path.join(gdir, f))]
    cnn_graphs = [{'file': f, 'title': t, 'desc': d} for f,(t,d) in cnn_map.items()
                   if os.path.exists(os.path.join(gdir, f))]
    cnn_acc_raw    = CNN_ARTIFACTS.get('test_accuracy', 0)
    cnn_acc        = round(float(cnn_acc_raw) * 100, 1) if cnn_acc_raw else None
    ens            = ML_ARTIFACTS.get('ensemble_results', {})
    ensemble_stats = [{'name': k, 'acc': round(v['acc']*100,1),
                        'auc': round(v['auc']*100,1), 'cv': round(v['cv']*100,1)}
                       for k,v in ens.items()]
    return render_template('dashboard.html',
        ml_graphs=ml_graphs, cnn_graphs=cnn_graphs,
        ml_acc=round(ML_ARTIFACTS.get('test_accuracy',0)*100,1),
        ml_auc=round(ML_ARTIFACTS.get('roc_auc',0)*100,1),
        cnn_acc=cnn_acc, cnn_model_ready=CNN_MODEL is not None,
        ensemble_stats=ensemble_stats,
    )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/debug/cnn')
def debug_cnn():
    """Debug route — lists every layer in the CNN model to diagnose Grad-CAM issues."""
    if CNN_MODEL is None:
        return "<pre>CNN model not loaded. Place cnn_model.keras in models/ and restart.</pre>", 404

    lines = ["<pre style='background:#0d1117;color:#e6edf3;padding:2rem;font-family:monospace'>"]
    lines.append("=== CNN MODEL LAYER INSPECTION ===\n")
    lines.append(f"Total top-level layers: {len(CNN_MODEL.layers)}\n\n")

    all_pairs = _collect_all_layers(CNN_MODEL)
    lines.append(f"Total layers (incl. nested): {len(all_pairs)}\n\n")
    lines.append(f"{'#':<5} {'Layer Name':<45} {'Type':<30} {'Output Shape'}\n")
    lines.append("-" * 110 + "\n")

    for i, (layer, src) in enumerate(all_pairs):
        ltype = type(layer).__name__
        try:
            shape = str(layer.output_shape)
        except Exception:
            shape = "N/A"
        marker = " ← CONV" if ('conv' in layer.name.lower() and len(shape) > 0) else ""
        lines.append(f"{i:<5} {layer.name:<45} {ltype:<30} {shape}{marker}\n")

    # Show which layer would be selected
    sub_model, layer_name, layer_idx = _find_submodel_and_layer(CNN_MODEL)
    lines.append(f"\n{'='*60}\n")
    if layer_name:
        lines.append(f"✅ Grad-CAM target layer : '{layer_name}'\n")
        lines.append(f"   Found in sub-model   : '{sub_model.name}'\n")
        lines.append(f"   Layer index in sub   : {layer_idx}\n")
    else:
        lines.append("❌ No suitable layer found — Grad-CAM cannot run\n")
    lines.append("</pre>")
    return "".join(lines)


# ══════════════════════════════════════════════
#  FUSION HELPER  (no model changes)
# ══════════════════════════════════════════════

# CNN class → severity index (0=none, 1=very mild, 2=mild, 3=moderate)
_CNN_SEVERITY_IDX = {
    'Non_Demented':       0,
    'Very_Mild_Demented': 1,
    'Mild_Demented':      2,
    'Moderate_Demented':  3,
}

def fuse_results(ml_prob_pos_pct, cnn_probs_dict, ml_weight=0.40, cnn_weight=0.60):
    """
    ml_prob_pos_pct : float — ML Alzheimer probability 0–100
    cnn_probs_dict  : dict  — {raw_class_key: probability_0_to_1}
    Returns         : dict  — final verdict fields
    """
    ml_prob  = ml_prob_pos_pct / 100.0

    # CNN: P(alzheimer) = 1 - P(Non_Demented)
    p_non    = cnn_probs_dict.get('Non_Demented', 0.0)
    cnn_alz  = 1.0 - p_non

    # CNN expected severity (0–3)
    cnn_sev  = sum(_CNN_SEVERITY_IDX.get(k, 0) * v for k, v in cnn_probs_dict.items())

    # ML severity proxy (0–3 scale)
    ml_sev   = ml_prob * 3.0

    # Combined
    combined_prob = ml_weight * ml_prob  + cnn_weight * cnn_alz
    combined_sev  = ml_weight * ml_sev   + cnn_weight * cnn_sev

    # Map combined severity → label/badge
    if combined_sev < 0.5:
        badge, label, icon = 'none',   "No Alzheimer's Detected",           'fa-circle-check'
    elif combined_sev < 1.3:
        badge, label, icon = 'low',    'Very Mild Cognitive Impairment',    'fa-circle-info'
    elif combined_sev < 2.2:
        badge, label, icon = 'medium', 'Mild Dementia Indicated',           'fa-triangle-exclamation'
    else:
        badge, label, icon = 'high',   'Moderate Dementia Detected',        'fa-radiation'

    risk_label = {'none': 'Low Risk', 'low': 'Low-Moderate Risk',
                  'medium': 'Moderate Risk', 'high': 'High Risk'}[badge]

    # Agreement between the two models
    ml_vote_alz  = ml_prob  >= 0.30
    cnn_vote_alz = cnn_alz  >= 0.30

    if ml_vote_alz == cnn_vote_alz:
        agree_cls, agree_icon, agree_text = 'agree-full',     'fa-handshake',           'Models Agree'
    elif abs(ml_prob - cnn_alz) < 0.25:
        agree_cls, agree_icon, agree_text = 'agree-partial',  'fa-circle-half-stroke',  'Partial Agreement'
    else:
        agree_cls, agree_icon, agree_text = 'agree-conflict', 'fa-triangle-exclamation','Models Conflict — Review'

    return {
        'badge':          badge,
        'label':          label,
        'icon':           icon,
        'combined_prob':  round(combined_prob * 100, 1),
        'severity_score': round(combined_sev, 2),
        'risk_label':     risk_label,
        'agree_cls':      agree_cls,
        'agree_icon':     agree_icon,
        'agree_text':     agree_text,
    }


# ══════════════════════════════════════════════
#  COMBINED ROUTES
# ══════════════════════════════════════════════

@app.route('/combined')
def combined_form():
    return render_template('combined_predict.html',
        cnn_ready = CNN_MODEL is not None,
        ml_ready  = bool(ML_MODELS),
    )


@app.route('/combined/predict', methods=['POST'])
def combined_predict():
    ml_data       = None
    cnn_data      = None
    cnn_probs_raw = {}
    errors        = []

    # ── 1. ML inference (same logic as ml_predict, no changes) ──
    if ML_MODELS:
        try:
            features     = ML_ARTIFACTS['feature_names']
            feature_vals = [float(request.form.get(f, '') or 0) for f in features]
            X            = np.array(feature_vals).reshape(1, -1)
            X_scaled     = ML_SCALER.transform(X)

            model_results, votes = [], []
            for name, model in ML_MODELS.items():
                proba    = model.predict_proba(X_scaled)[0]
                pred     = int(np.argmax(proba))
                votes.append(pred)
                ens_info = ML_ARTIFACTS.get('ensemble_results', {}).get(name, {})
                prob_pos = round(float(proba[1]) * 100, 2)
                model_results.append({
                    'name':       name,
                    'prediction': pred,
                    'label':      'Alzheimer' if pred == 1 else 'No Alzheimer',
                    'prob_pos':   prob_pos,
                    'prob_neg':   round(100 - prob_pos, 2),
                    'confidence': round(float(max(proba)) * 100, 2),
                    'test_acc':   round(float(ens_info.get('acc', 0)) * 100, 1),
                    'auc':        round(float(ens_info.get('auc', 0)) * 100, 1),
                })

            final_vote   = int(np.bincount(votes).argmax())
            avg_prob_pos = round(float(np.mean([r['prob_pos'] for r in model_results])), 2)
            risk, risk_lvl, risk_color = risk_meta(avg_prob_pos / 100)

            ml_data = {
                'final_label':   'Alzheimer Detected' if final_vote == 1 else 'No Alzheimer Detected',
                'final_vote':    final_vote,
                'avg_prob_pos':  avg_prob_pos,
                'avg_prob_neg':  round(100 - avg_prob_pos, 2),
                'risk_lvl':      risk_lvl,
                'risk_color':    risk_color,
                'vote_count':    votes.count(final_vote),
                'total_models':  len(ML_MODELS),
                'model_results': model_results,
            }
        except Exception as e:
            errors.append(f'ML error: {e}')

    # ── 2. CNN inference (same logic as cnn_predict, no changes) ──
    if CNN_MODEL and 'mri_image' in request.files and request.files['mri_image'].filename != '':
        try:
            file      = request.files['mri_image']
            base, ext = os.path.splitext(secure_filename(file.filename))
            filename  = f"{base}_{int(time.time())}{ext}"
            filepath  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            cleanup_old_uploads()

            img_size = CNN_ARTIFACTS.get('img_size', (224, 224))
            img_arr  = np.expand_dims(
                np.array(Image.open(filepath).convert('RGB').resize(img_size),
                         dtype=np.float32), 0
            )
            preds         = CNN_MODEL.predict(img_arr, verbose=0)[0]
            class_indices = CNN_ARTIFACTS['class_indices']
            idx_to_class  = {v: k for k, v in class_indices.items()}
            pred_idx      = int(np.argmax(preds))
            pred_class    = idx_to_class.get(pred_idx, 'Unknown')
            severity, badge = CNN_SEVERITY.get(pred_class, ('Unknown', 'medium'))

            _class_color = {
                'Non_Demented':       ('var(--green)', 'prog-green'),
                'Very_Mild_Demented': ('var(--cyan)',  'prog-cyan'),
                'Mild_Demented':      ('var(--amber)', 'prog-yellow'),
                'Moderate_Demented':  ('var(--red)',   'prog-red'),
            }
            all_probs = sorted([
                {'class':     CNN_DISPLAY.get(idx_to_class.get(i, ''), f'Class {i}'),
                 'prob':      round(float(preds[i]) * 100, 2),
                 'raw_key':   idx_to_class.get(i, f'class_{i}'),
                 'txt_color': _class_color.get(idx_to_class.get(i, ''),
                                               ('var(--text)', 'prog-cyan'))[0],
                 'bar_color': _class_color.get(idx_to_class.get(i, ''),
                                               ('var(--text)', 'prog-cyan'))[1]}
                for i in range(len(preds))
            ], key=lambda x: x['prob'], reverse=True)

            for i in range(len(preds)):
                raw_key = idx_to_class.get(i, f'class_{i}')
                cnn_probs_raw[raw_key] = float(preds[i])

            cnn_data = {
                'pred_class':  CNN_DISPLAY.get(pred_class, pred_class),
                'severity':    severity,
                'badge':       badge,
                'confidence':  round(float(preds[pred_idx]) * 100, 2),
                'all_probs':   all_probs,
                'img_url':     url_for('static', filename=f'uploads/{filename}'),
            }
        except Exception as e:
            errors.append(f'CNN error: {e}')

    # ── 3. Fuse → final conclusion ──────────────────────────────
    if ml_data and cnn_data:
        final = fuse_results(ml_data['avg_prob_pos'], cnn_probs_raw)
    elif ml_data:
        p = ml_data['avg_prob_pos'] / 100
        risk, _, _ = risk_meta(p)
        sev = p * 3
        if sev < 0.5:    badge = 'none'
        elif sev < 1.3:  badge = 'low'
        elif sev < 2.2:  badge = 'medium'
        else:             badge = 'high'
        final = {
            'badge':          badge,
            'label':          ml_data['final_label'],
            'icon':           'fa-circle-check' if ml_data['final_vote'] == 0 else 'fa-triangle-exclamation',
            'combined_prob':  ml_data['avg_prob_pos'],
            'severity_score': round(p * 3, 2),
            'risk_label':     risk,
            'agree_cls':      'agree-partial',
            'agree_icon':     'fa-circle-half-stroke',
            'agree_text':     'Clinical Only (No MRI)',
        }
    elif cnn_data:
        badge      = cnn_data['badge']
        cnn_alz_p  = round((1 - cnn_probs_raw.get('Non_Demented', 0)) * 100, 1)
        sev_idx    = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[badge]
        icon_map   = {'none': 'fa-circle-check', 'low': 'fa-circle-info',
                      'medium': 'fa-triangle-exclamation', 'high': 'fa-radiation'}
        final = {
            'badge':          badge,
            'label':          cnn_data['severity'],
            'icon':           icon_map[badge],
            'combined_prob':  cnn_alz_p,
            'severity_score': round(float(sev_idx), 2),
            'risk_label':     {'none':'Low Risk','low':'Low-Moderate Risk',
                               'medium':'Moderate Risk','high':'High Risk'}[badge],
            'agree_cls':      'agree-partial',
            'agree_icon':     'fa-circle-half-stroke',
            'agree_text':     'MRI Only (No Clinical)',
        }
    else:
        final = {
            'badge': 'none', 'label': 'Analysis Incomplete', 'icon': 'fa-circle-xmark',
            'combined_prob': 0, 'severity_score': 0, 'risk_label': 'N/A',
            'agree_cls': 'agree-conflict', 'agree_icon': 'fa-circle-xmark', 'agree_text': 'No Data',
        }

    return render_template('combined_result.html',
        ml=ml_data, cnn=cnn_data, final=final, errors=errors,
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
