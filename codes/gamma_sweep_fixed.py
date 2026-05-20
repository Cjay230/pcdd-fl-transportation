# gamma_sweep_fixed.py
# Replicates the EXACT ablation_60s.py setup (seed, data, MLP, helpers) so
# that gamma=0.2 produces the same F1=0.6313 as the original E5.
#
# Strategy:
#   1. Set seed=42 and load data exactly as ablation_60s.py does.
#   2. Run E1-E4 in full (including evaluate() calls that consume RNG) so
#      the RNG reaches the same state it was at when E5 started.
#   3. Capture that RNG state.
#   4. For each gamma: restore the captured state, init a fresh model, run
#      the E5 training loop (CW loss + KI + cwFedAvg + LAWA).
#   5. gamma=0.2 must match 0.6313.

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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WIN_DIR   = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"
SAVE_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
PLOTS_DIR = r"C:\Users\user\Desktop\electives\VIP\PyTorch\plots"
os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── exact constants from ablation_60s.py ──────────────────────
CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
ALL_CLASSES = list(range(N_CLASSES))

# ── seed exactly as ablation_60s.py ───────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# DATA LOADING — exact copy from ablation_60s.py
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("LOADING 60s WINDOWED DATA")
print("=" * 65)

city_dfs = {}
for city in CITIES:
    path = os.path.join(WIN_DIR, f'{city}_60s.csv')
    city_dfs[city] = pd.read_csv(path)
    print(f"  {city:8s}: {len(city_dfs[city])} windows")

sample_df  = city_dfs['jeddah']
feat_cols  = [c for c in sample_df.columns if c != 'label']
N_FEATURES = len(feat_cols)
print(f"  Features: {N_FEATURES}\n")

X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
city_classes = {}

for city in CITIES:
    df = city_dfs[city].copy()
    df['label_enc'] = df['label'].map(LABEL_MAP).fillna(-1).astype(np.int64)
    df = df[df['label_enc'] >= 0]
    X = df[feat_cols].fillna(0).values.astype(np.float32)
    y = df['label_enc'].values
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                random_state=42)
    X_train_d[city] = Xtr
    X_test_d[city]  = Xte
    y_train_d[city] = ytr.astype(np.int64)
    y_test_d[city]  = yte.astype(np.int64)
    city_classes[city] = set(int(c) for c in np.unique(ytr))

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"  Global test: {len(y_test_global)} samples")

# ─────────────────────────────────────────────────────────────
# MODEL — exact copy (no argument, uses global N_FEATURES)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# HELPERS — exact copies from ablation_60s.py
# ─────────────────────────────────────────────────────────────
def get_state(model):    return copy.deepcopy(model.state_dict())
def set_state(model, s): model.load_state_dict(copy.deepcopy(s))

def class_count_vec(y):
    v = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        v[c] = (y == c).sum()
    return v

def evaluate(global_state):
    m = MLP()          # consumes RNG — must replicate for E1-E4
    set_state(m, global_state)
    m.eval()
    with torch.no_grad():
        preds = m(torch.tensor(X_test_global)).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test_global, preds))
    g_f1  = float(f1_score(y_test_global, preds, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    pc_f1 = f1_score(y_test_global, preds, average=None,
                     labels=ALL_CLASSES, zero_division=0)
    city_m = {}
    for city in CITIES:
        with torch.no_grad():
            yp = m(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        city_m[city] = float(f1_score(y_test_d[city], yp, average='macro',
                                      labels=ALL_CLASSES, zero_division=0))
    return g_acc, g_f1, pc_f1, city_m

def train_local(global_state, city, use_cw_loss=False, use_ki=False):
    model = MLP()
    set_state(model, global_state)
    model.train()
    if use_cw_loss:
        y_arr   = y_train_d[city]
        n_tot   = len(y_arr)
        weights = torch.zeros(N_CLASSES)
        for c in range(N_CLASSES):
            n_c = (y_arr == c).sum()
            if n_c > 0:
                weights[c] = n_tot / (N_CLASSES * n_c)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()
    if use_ki:
        missing = [c for c in range(N_CLASSES) if c not in city_classes[city]]
        if missing:
            with torch.no_grad():
                g_w = global_state['fc_out.weight']
                g_b = global_state['fc_out.bias']
                for c in missing:
                    model.fc_out.weight[c].copy_(g_w[c])
                    model.fc_out.bias[c].copy_(g_b[c])
    return model

def agg_fedavg(local_models):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w = sizes / sizes.sum()
    new_state = {}
    for key in local_models[0].state_dict():
        new_state[key] = sum(w[k] * local_models[k].state_dict()[key].float()
                             for k in range(len(CITIES)))
    return new_state

def agg_cw(local_models):
    n     = len(local_models)
    cc    = np.array([class_count_vec(y_train_d[c]) for c in CITIES])
    sizes = cc.sum(axis=1)
    total = sizes.sum()
    w_scalar = sizes / total
    ct   = cc.sum(axis=0)
    w_cw = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)
    new_state = {}
    for key in local_models[0].state_dict():
        if 'fc_out' in key:
            continue
        new_state[key] = sum(w_scalar[k] * local_models[k].state_dict()[key].float()
                             for k in range(n))
    out_w = torch.zeros_like(local_models[0].state_dict()['fc_out.weight'])
    out_b = torch.zeros_like(local_models[0].state_dict()['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += w_cw[k,c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += w_cw[k,c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b
    return new_state

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

def agg_cw_lawa(local_models, gamma):
    n     = len(local_models)
    cc    = np.array([class_count_vec(y_train_d[c]) for c in CITIES])
    sizes = cc.sum(axis=1)
    total = sizes.sum()
    w_scalar = sizes / total
    ct   = cc.sum(axis=0)
    w_cw = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)
    lawa_w = compute_lawa_weights(local_models)
    bl_s  = gamma * lawa_w + (1 - gamma) * w_scalar
    bl_s /= bl_s.sum()
    bl_cw = gamma * lawa_w[:, None] + (1 - gamma) * w_cw
    for c in range(N_CLASSES):
        s = bl_cw[:, c].sum()
        if s > 0:
            bl_cw[:, c] /= s
    new_state = {}
    for key in local_models[0].state_dict():
        if 'fc_out' in key:
            continue
        new_state[key] = sum(bl_s[k] * local_models[k].state_dict()[key].float()
                             for k in range(n))
    out_w = torch.zeros_like(local_models[0].state_dict()['fc_out.weight'])
    out_b = torch.zeros_like(local_models[0].state_dict()['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += bl_cw[k,c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += bl_cw[k,c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b
    return new_state

# ─────────────────────────────────────────────────────────────
# STEP 1 — Run E1-E4 to advance RNG to where E5 started
# (silent: no printing of results, just consume the same RNG ops)
# ─────────────────────────────────────────────────────────────
print("\nAdvancing RNG through E1-E4 (replicating ablation sequence)...")

_e1e4 = [
    dict(use_cw_loss=False, use_ki=False, agg_fn=agg_fedavg),  # E1
    dict(use_cw_loss=True,  use_ki=False, agg_fn=agg_fedavg),  # E2
    dict(use_cw_loss=True,  use_ki=True,  agg_fn=agg_fedavg),  # E3
    dict(use_cw_loss=True,  use_ki=True,  agg_fn=agg_cw),      # E4
]

for ei, cfg in enumerate(_e1e4, 1):
    gs = get_state(MLP())
    for rnd in range(1, N_ROUNDS + 1):
        lm = [train_local(gs, city, use_cw_loss=cfg['use_cw_loss'],
                           use_ki=cfg['use_ki'])
              for city in CITIES]
        gs = cfg['agg_fn'](lm)
        evaluate(gs)   # must call — creates MLP() each time, consuming RNG
    print(f"  E{ei} done.")

# ─────────────────────────────────────────────────────────────
# STEP 2 — Capture RNG state (this is where E5 starts in the original)
# ─────────────────────────────────────────────────────────────
rng_torch = torch.get_rng_state()
rng_np    = np.random.get_state()
print("RNG state captured — beginning gamma sweep.\n")

# ─────────────────────────────────────────────────────────────
# STEP 3 — Gamma sweep: restore state, run E5 for each gamma
# ─────────────────────────────────────────────────────────────
gammas  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
f1_vals = []

print("=" * 65)
print("GAMMA SWEEP — E5 method (CW loss + KI + cwFedAvg + LAWA)")
print("=" * 65)

for gamma in gammas:
    torch.set_rng_state(rng_torch)
    np.random.set_state(rng_np)

    global_state = get_state(MLP())
    for rnd in range(1, N_ROUNDS + 1):
        local_models = [
            train_local(global_state, city, use_cw_loss=True, use_ki=True)
            for city in CITIES
        ]
        global_state = agg_cw_lawa(local_models, gamma)
        evaluate(global_state)   # keep RNG consistent across gammas

    g_acc, g_f1, pc_f1, city_m = evaluate(global_state)
    f1_vals.append(g_f1)

    check = " <-- must be 0.6313" if gamma == 0.2 else ""
    print(f"  gamma={gamma:.1f}  Acc={g_acc:.4f}  Macro F1={g_f1:.4f}{check}")

# ─────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────
lines = [
    "Gamma Sweep — E5 method (CW loss + KI + cwFedAvg + LAWA)\n",
    "Fixed seed=42, RNG advanced through E1-E4 to match original ablation\n",
    f"\n{'gamma':>8}  {'Macro F1':>10}\n",
    "-"*22 + "\n",
]
for g, f in zip(gammas, f1_vals):
    lines.append(f"{g:>8.1f}  {f:>10.4f}\n")
with open(os.path.join(SAVE_DIR, "gamma_sweep.txt"), "w") as fh:
    fh.writelines(lines)

# ─────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(gammas, f1_vals, marker='o', linewidth=2, markersize=8,
        color='steelblue', label='E5 (full method)')
ax.set_xlabel('Gamma (γ)', fontsize=12)
ax.set_ylabel('Global Macro F1 (Round 10)', fontsize=12)
ax.set_title('Macro F1 vs. Gamma — E5 Method, 60s Windows', fontsize=13)
ax.set_xticks(gammas)
ax.set_ylim(min(f1_vals) - 0.02, max(f1_vals) + 0.03)
ax.grid(True, alpha=0.4)
ax.legend()
for g, f in zip(gammas, f1_vals):
    ax.annotate(f'{f:.4f}', (g, f), textcoords='offset points',
                xytext=(0, 9), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_gamma_sweep.png"), dpi=150)
plt.close()

print(f"\nSaved: results/gamma_sweep.txt, plots/11_gamma_sweep.png")
print("Done.")
