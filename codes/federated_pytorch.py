# Federated Learning - PyTorch MLP on 300-second windowed data
# Experiment 1 : Standard FedAvg
# Experiment 2a: cwFedAvg + LAWA (gamma=0.2)
# Experiment 2b: cwFedAvg + LAWA (gamma=0.3)
# Experiment 2c: cwFedAvg + LAWA (gamma=0.1)
import os, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
WINDOWED_DIR = r"C:\Users\user\Desktop\electives\VIP\kaust data\windowed"
SAVE_DIR     = r"C:\Users\user\Desktop\electives\VIP\PyTorch"
RESULTS_DIR  = os.path.join(SAVE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CITIES    = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP = {'car': 0, 'walk': 1, 'bus': 2, 'scooter': 3,
             'bike': 4, 'motorcycle': 5, 'jog': 6}
CLASS_NAMES = ['car', 'walk', 'bus', 'scooter', 'bike', 'motorcycle', 'jog']
N_CLASSES   = 7
N_FEATURES  = 29
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9

META_COLS = ['label', 'source_file', 'window_id', 't_start_sec', 't_end_sec']

torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 256), nn.ReLU(),
            nn.Linear(256, 256),        nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        return self.fc_out(self.net(x))


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING & SPLITTING DATA")
print("=" * 65)

feat_cols = None
X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
city_classes = {}   # classes present per city (as set of ints)

raw_dfs = {}
for city in CITIES:
    df = pd.read_csv(f'{WINDOWED_DIR}/{city}_windowed.csv')
    raw_dfs[city] = df

# Determine feature columns (same for all cities)
feat_cols = [c for c in raw_dfs['jeddah'].columns if c not in META_COLS]
N_FEATURES = len(feat_cols)
print(f"Feature columns ({N_FEATURES}): {feat_cols}\n")

for city in CITIES:
    df = raw_dfs[city].copy()
    df['label_enc'] = df['label'].map(LABEL_MAP)

    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values.astype(np.int64)

    present = sorted(df['label_enc'].unique().tolist())
    city_classes[city] = set(present)

    # Stratified split; fall back to random if a class has <2 samples
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42)

    X_train_d[city] = Xtr
    X_test_d[city]  = Xte
    y_train_d[city] = ytr
    y_test_d[city]  = yte

    dist = {CLASS_NAMES[c]: int((y == c).sum()) for c in present}
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}  "
          f"classes={[CLASS_NAMES[c] for c in present]}")
    print(f"            dist={dist}")

# Global scaler fit on all training data
scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"\n  Global test set: {len(y_test_global)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def class_count_vec(y):
    """Count per-class samples. Returns numpy array shape (N_CLASSES,)."""
    v = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        v[c] = (y == c).sum()
    return v

def get_state(model):
    return copy.deepcopy(model.state_dict())

def set_state(model, state):
    model.load_state_dict(copy.deepcopy(state))

def evaluate(model):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test_global)
        preds = model(Xt).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test_global, preds))
    g_f1  = float(f1_score(y_test_global, preds, average='macro',
                            labels=list(range(N_CLASSES)), zero_division=0))
    pc_f1 = f1_score(y_test_global, preds, average=None,
                     labels=list(range(N_CLASSES)), zero_division=0)
    city_m = {}
    for city in CITIES:
        Xc = torch.tensor(X_test_d[city])
        yc = y_test_d[city]
        pc = model(Xc).argmax(dim=1).numpy()
        city_m[city] = {
            'acc': float(accuracy_score(yc, pc)),
            'f1':  float(f1_score(yc, pc, average='macro',
                                  labels=list(range(N_CLASSES)), zero_division=0))
        }
    return g_acc, g_f1, pc_f1, city_m


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING  (Experiment 1 — standard, no class weights)
# ─────────────────────────────────────────────────────────────────────────────
def train_local_standard(global_state, city):
    model = MLP()
    set_state(model, global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    X = torch.tensor(X_train_d[city])
    y = torch.tensor(y_train_d[city])
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()

    return model


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRAINING  (Experiment 2 — class-weighted loss + knowledge inheritance)
# ─────────────────────────────────────────────────────────────────────────────
def train_local_advanced(global_state, city):
    model = MLP()
    set_state(model, global_state)
    model.train()

    # Class weights: total / (n_classes × n_c);  0 for missing classes
    y_arr = y_train_d[city]
    n_total = len(y_arr)
    weights = torch.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        n_c = (y_arr == c).sum()
        if n_c > 0:
            weights[c] = n_total / (N_CLASSES * n_c)
        # else stays 0

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    X = torch.tensor(X_train_d[city])
    y = torch.tensor(y_arr)
    loader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()

    # Knowledge inheritance: restore global output-node weights for missing classes
    missing = [c for c in range(N_CLASSES) if c not in city_classes[city]]
    if missing:
        with torch.no_grad():
            g_w = global_state['fc_out.weight']
            g_b = global_state['fc_out.bias']
            for c in missing:
                model.fc_out.weight[c].copy_(g_w[c])
                model.fc_out.bias[c].copy_(g_b[c])

    return model


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION — FedAvg
# ─────────────────────────────────────────────────────────────────────────────
def agg_fedavg(local_models):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w     = sizes / sizes.sum()

    new_state = {}
    ref_state = local_models[0].state_dict()
    for key in ref_state:
        new_state[key] = sum(
            w[k] * local_models[k].state_dict()[key].float()
            for k in range(len(CITIES))
        )
    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION — cwFedAvg + LAWA blend
# ─────────────────────────────────────────────────────────────────────────────
def compute_lawa_weights(local_models):
    n     = len(local_models)
    raw_w = np.zeros(n)
    for k, model in enumerate(local_models):
        city = CITIES[k]
        model.eval()
        with torch.no_grad():
            logits   = model(torch.tensor(X_train_d[city]))
            log_prob = torch.log_softmax(logits, dim=1).numpy()
        y = y_train_d[city]
        losses = []
        for c in range(N_CLASSES):
            mask = (y == c)
            if mask.sum() > 0:
                losses.append(float(-log_prob[mask, c].mean()))
        if len(losses) < 2:
            raw_w[k] = 1.0
        else:
            L_min, L_max = min(losses), max(losses)
            raw_w[k] = 1.0 if L_max == 0 else float(L_min / L_max)
    s = raw_w.sum()
    return raw_w / s if s > 0 else np.ones(n) / n


def agg_blended(local_models, gamma):
    n      = len(local_models)
    cc     = np.array([class_count_vec(y_train_d[c]) for c in CITIES])  # (n,7)
    sizes  = cc.sum(axis=1)
    total  = sizes.sum()

    w_scalar = sizes / total                    # (n,) standard weights

    ct   = cc.sum(axis=0)                        # (7,)
    w_cw = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)  # (n,7)

    lawa_w = compute_lawa_weights(local_models)  # (n,) normalised

    # Blended scalar (hidden layers)
    bl_s = gamma * lawa_w + (1 - gamma) * w_scalar
    bl_s /= bl_s.sum()

    # Blended class-wise (output layer)
    bl_cw = gamma * lawa_w[:, None] + (1 - gamma) * w_cw   # (n,7)
    for c in range(N_CLASSES):
        s = bl_cw[:, c].sum()
        if s > 0:
            bl_cw[:, c] /= s

    # Build new state dict
    ref_state = local_models[0].state_dict()
    new_state = {}

    for key in ref_state:
        if 'fc_out' in key:
            continue   # handled separately below
        new_state[key] = sum(
            bl_s[k] * local_models[k].state_dict()[key].float()
            for k in range(n)
        )

    # Output layer — weight (N_CLASSES, 256) and bias (N_CLASSES,)
    out_w = torch.zeros_like(ref_state['fc_out.weight'])
    out_b = torch.zeros_like(ref_state['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += bl_cw[k, c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += bl_cw[k, c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b

    return new_state


# ─────────────────────────────────────────────────────────────────────────────
# FEDERATED TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(exp_name, train_fn, agg_fn, save_path):
    print(f"\n{'='*65}")
    print(f"  {exp_name}")
    print(f"{'='*65}")

    global_model = MLP()
    global_state = get_state(global_model)
    rows = []
    t0_exp = time.time()

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()

        # Local training
        local_models = []
        for city in CITIES:
            m = train_fn(global_state, city)
            local_models.append(m)

        # Aggregation
        global_state = agg_fn(local_models)
        set_state(global_model, global_state)

        # Evaluation
        g_acc, g_f1, pc_f1, city_m = evaluate(global_model)
        elapsed = time.time() - t0

        print(f"  Round {rnd:2d}/10  "
              f"GlobalAcc {g_acc:.4f}  GlobalMacroF1 {g_f1:.4f}  "
              f"({elapsed:.1f}s)")

        row = {
            'experiment':      exp_name,
            'round':           rnd,
            'global_acc':      round(g_acc, 6),
            'global_macro_f1': round(g_f1,  6),
        }
        for city in CITIES:
            row[f'{city}_acc']      = round(city_m[city]['acc'], 6)
            row[f'{city}_macro_f1'] = round(city_m[city]['f1'],  6)
        for i, cls in enumerate(CLASS_NAMES):
            row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    total_min = (time.time() - t0_exp) / 60
    print(f"  Finished in {total_min:.1f} min  ->  {save_path}")

    # Print round-10 summary
    last = df[df['round'] == N_ROUNDS].iloc[0]
    print(f"\n  --- Round 10 Results ---")
    print(f"  Global Accuracy : {last['global_acc']:.4f}")
    print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
    print(f"  Per-city Macro F1:")
    for city in CITIES:
        print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
    print(f"  Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RUN EXPERIMENT 1 — Standard FedAvg
# ─────────────────────────────────────────────────────────────────────────────
df1 = run_experiment(
    "FedAvg (baseline)",
    train_fn = train_local_standard,
    agg_fn   = agg_fedavg,
    save_path= os.path.join(RESULTS_DIR, "fedavg_windowed.csv")
)

print("\nExperiment 1 done. Proceeding to Experiment 2 (cwFedAvg+LAWA)...")

# ─────────────────────────────────────────────────────────────────────────────
# RUN EXPERIMENT 2 — cwFedAvg + LAWA (gamma = 0.2, 0.3, 0.1)
# ─────────────────────────────────────────────────────────────────────────────
exp2_results = []

for gamma, fname in [(0.2, "cwfedavg_lawa_gamma02.csv"),
                     (0.3, "cwfedavg_lawa_gamma03.csv"),
                     (0.1, "cwfedavg_lawa_gamma01.csv")]:

    df_g = run_experiment(
        f"cwFedAvg+LAWA(gamma={gamma})",
        train_fn = train_local_advanced,
        agg_fn   = lambda models, g=gamma: agg_blended(models, g),
        save_path= os.path.join(RESULTS_DIR, fname)
    )
    exp2_results.append(df_g)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL COMPARISON TABLE (Round 10 of all experiments)
# ─────────────────────────────────────────────────────────────────────────────
all_dfs = [df1] + exp2_results
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(os.path.join(RESULTS_DIR, "all_experiments.csv"), index=False)

last_rounds = combined[combined['round'] == N_ROUNDS].copy()

# Add sklearn baseline for reference
sklearn_ref = {
    'experiment': 'sklearn FedAvg (raw rows baseline)',
    'global_acc': 0.4139, 'global_macro_f1': 0.2869,
    **{f'{c}_acc': '-' for c in CITIES},
    **{f'{c}_macro_f1': '-' for c in CITIES},
    **{f'f1_{cls}': '-' for cls in CLASS_NAMES},
    'round': 10
}
ref_df = pd.DataFrame([sklearn_ref])
last_rounds = pd.concat([last_rounds, ref_df], ignore_index=True)

print(f"\n{'='*65}")
print("FINAL COMPARISON TABLE — Round 10")
print(f"{'='*65}")

print("\n--- Global Metrics ---")
print(last_rounds[['experiment', 'global_acc', 'global_macro_f1']].to_string(index=False))

print("\n--- Per-City Macro F1 ---")
print(last_rounds[['experiment'] + [f'{c}_macro_f1' for c in CITIES]].to_string(index=False))

print("\n--- Per-Class F1 ---")
print(last_rounds[['experiment'] + [f'f1_{cls}' for cls in CLASS_NAMES]].to_string(index=False))

print(f"\nAll results saved to: {RESULTS_DIR}")
print("Done.")
