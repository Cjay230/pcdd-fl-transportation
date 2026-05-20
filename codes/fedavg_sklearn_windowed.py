# FedAvg on windowed data using sklearn MLPClassifier (no PyTorch)
import os, copy, time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

WINDOWED_DIR = r"C:\Users\user\Desktop\electives\VIP\kaust data\windowed"
RESULTS_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CITIES     = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP  = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3,
              'bike': 4, 'motorcycle': 5, 'jog': 6}
CLASS_NAMES = ['car', 'walk', 'bus', 'scooter', 'bike', 'motorcycle', 'jog']
N_CLASSES   = 7
N_ROUNDS    = 10
META_COLS   = ['label', 'source_file', 'window_id', 't_start_sec', 't_end_sec']

np.random.seed(42)

# ── Load data ────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING & SPLITTING DATA")
print("=" * 60)

X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}

sample_df = pd.read_csv(f'{WINDOWED_DIR}/jeddah_windowed.csv')
feat_cols  = [c for c in sample_df.columns if c not in META_COLS]
print(f"Features ({len(feat_cols)}): {feat_cols}\n")

for city in CITIES:
    df = pd.read_csv(f'{WINDOWED_DIR}/{city}_windowed.csv')
    df['label_enc'] = df['label'].map(LABEL_MAP)
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values.astype(np.int32)
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_d[city], X_test_d[city] = Xtr, Xte
    y_train_d[city], y_test_d[city] = ytr, yte
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}")

# Scale
scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c] for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"\n  Global test set: {len(y_test_global)} samples")

# ── Helpers ──────────────────────────────────────────────────────────────────
ALL_CLASSES = list(range(N_CLASSES))

def get_weights(mlp):
    return ([w.copy() for w in mlp.coefs_],
            [b.copy() for b in mlp.intercepts_])

def set_weights(mlp, coefs, intercepts):
    mlp.coefs_      = [w.copy() for w in coefs]
    mlp.intercepts_ = [b.copy() for b in intercepts]

def make_template(n_features):
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        solver='sgd',
        learning_rate_init=0.01,
        momentum=0.9,
        batch_size=32,
        max_iter=1,
        random_state=42
    )
    X_seed = np.zeros((N_CLASSES, n_features), dtype=np.float32)
    y_seed = np.arange(N_CLASSES, dtype=np.int32)
    mlp.partial_fit(X_seed, y_seed, classes=ALL_CLASSES)
    return mlp

def train_client(template, g_coefs, g_biases, X_tr, y_tr):
    local = copy.deepcopy(template)
    set_weights(local, g_coefs, g_biases)
    local.partial_fit(X_tr, y_tr)
    return local

def fedavg(models):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w     = sizes / sizes.sum()
    n_lay = len(models[0].coefs_)
    coefs  = [sum(w[k] * models[k].coefs_[l]      for k in range(len(CITIES))) for l in range(n_lay)]
    biases = [sum(w[k] * models[k].intercepts_[l] for k in range(len(CITIES))) for l in range(n_lay)]
    return coefs, biases

def evaluate(template):
    y_pred = template.predict(X_test_global)
    g_acc  = float(accuracy_score(y_test_global, y_pred))
    g_f1   = float(f1_score(y_test_global, y_pred, average='macro',
                             labels=ALL_CLASSES, zero_division=0))
    pc_f1  = f1_score(y_test_global, y_pred, average=None,
                      labels=ALL_CLASSES, zero_division=0)
    city_m = {}
    for city in CITIES:
        yp = template.predict(X_test_d[city])
        city_m[city] = {
            'acc': float(accuracy_score(y_test_d[city], yp)),
            'f1':  float(f1_score(y_test_d[city], yp, average='macro',
                                  labels=ALL_CLASSES, zero_division=0))
        }
    return g_acc, g_f1, pc_f1, city_m

# ── Run ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FedAvg — sklearn MLP — windowed data")
print("=" * 60)

n_features = len(feat_cols)
template   = make_template(n_features)
g_coefs, g_biases = get_weights(template)
rows = []

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()
    locals_ = [train_client(template, g_coefs, g_biases,
                            X_train_d[c], y_train_d[c]) for c in CITIES]
    g_coefs, g_biases = fedavg(locals_)
    set_weights(template, g_coefs, g_biases)

    g_acc, g_f1, pc_f1, city_m = evaluate(template)
    print(f"  Round {rnd:2d}/10  GlobalAcc {g_acc:.4f}  GlobalMacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'round': rnd, 'global_acc': round(g_acc,6), 'global_macro_f1': round(g_f1,6)}
    for city in CITIES:
        row[f'{city}_acc']      = round(city_m[city]['acc'], 6)
        row[f'{city}_macro_f1'] = round(city_m[city]['f1'],  6)
    for i, cls in enumerate(CLASS_NAMES):
        row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
    rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(RESULTS_DIR, 'fedavg_sklearn_windowed.csv')
df.to_csv(out, index=False)

# ── Summary ──────────────────────────────────────────────────────────────────
last = df[df['round'] == N_ROUNDS].iloc[0]
print(f"\n--- Round 10 Results ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
print(f"  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")
print(f"\nSaved -> {out}")
