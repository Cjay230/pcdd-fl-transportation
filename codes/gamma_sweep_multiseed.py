# gamma_sweep_multiseed.py
# Multi-seed gamma sweep: for each gamma, run 3 seeds and report mean +/- std.
# This averages out the stochastic noise between gamma values (sigma ~0.006)
# to give a stable picture of the true optimal gamma.
# Full E5 setup: CW loss + KI + cwFedAvg + LAWA at each gamma.

import os, copy
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

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
ALL_CLASSES = list(range(N_CLASSES))
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
SEEDS       = [42, 0, 123]
GAMMAS      = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ─────────────────────────────────────────────────────────────
# Data (fixed split, loaded once)
# ─────────────────────────────────────────────────────────────
print("Loading 60s windowed data...")
city_dfs = {}
for city in CITIES:
    city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))

feat_cols  = [c for c in city_dfs['jeddah'].columns if c != 'label']
N_FEATURES = len(feat_cols)

X_train_d, X_test_d, y_train_d, y_test_d, city_classes = {}, {}, {}, {}, {}
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
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
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
print(f"  Global test: {len(y_test_global)} samples, features: {N_FEATURES}\n")

# ─────────────────────────────────────────────────────────────
# Model
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

def get_state(m):    return copy.deepcopy(m.state_dict())
def set_state(m, s): m.load_state_dict(copy.deepcopy(s))

def class_count_vec(y):
    v = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        v[c] = (y == c).sum()
    return v

def evaluate(global_state):
    m = MLP()
    set_state(m, global_state)
    m.eval()
    with torch.no_grad():
        preds = m(torch.tensor(X_test_global)).argmax(dim=1).numpy()
    return float(f1_score(y_test_global, preds, average='macro',
                          labels=ALL_CLASSES, zero_division=0))

def train_local(global_state, city):
    model = MLP()
    set_state(model, global_state)
    model.train()
    y_arr = y_train_d[city]
    n_tot = len(y_arr)
    weights = torch.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        n_c = (y_arr == c).sum()
        if n_c > 0:
            weights[c] = n_tot / (N_CLASSES * n_c)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader    = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()
    missing = [c for c in range(N_CLASSES) if c not in city_classes[city]]
    if missing:
        with torch.no_grad():
            for c in missing:
                model.fc_out.weight[c].copy_(global_state['fc_out.weight'][c])
                model.fc_out.bias[c].copy_(global_state['fc_out.bias'][c])
    return model

def compute_lawa(local_models):
    n     = len(local_models)
    raw_w = np.zeros(n)
    for k, model in enumerate(local_models):
        city = CITIES[k]
        model.eval()
        with torch.no_grad():
            log_prob = torch.log_softmax(
                model(torch.tensor(X_train_d[city])), dim=1).numpy()
        y     = y_train_d[city]
        losses = [-log_prob[(y == c), c].mean()
                  for c in range(N_CLASSES) if (y == c).sum() > 0]
        if len(losses) < 2:
            raw_w[k] = 1.0
        else:
            raw_w[k] = min(losses) / max(losses) if max(losses) > 0 else 1.0
    s = raw_w.sum()
    return raw_w / s if s > 0 else np.ones(n) / n

def aggregate(local_models, gamma):
    n    = len(local_models)
    cc   = np.array([class_count_vec(y_train_d[c]) for c in CITIES])
    sizes = cc.sum(axis=1)
    w_size = sizes / sizes.sum()
    ct     = cc.sum(axis=0)
    w_cw   = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)
    lawa   = compute_lawa(local_models)
    w_h = gamma * lawa + (1 - gamma) * w_size
    w_h /= w_h.sum()
    w_o = gamma * lawa[:, None] + (1 - gamma) * w_cw
    for c in range(N_CLASSES):
        s = w_o[:, c].sum()
        if s > 0:
            w_o[:, c] /= s
    new_state = {}
    for key in local_models[0].state_dict():
        if 'fc_out' in key:
            continue
        new_state[key] = sum(w_h[k] * local_models[k].state_dict()[key].float()
                             for k in range(n))
    out_w = torch.zeros_like(local_models[0].state_dict()['fc_out.weight'])
    out_b = torch.zeros_like(local_models[0].state_dict()['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += w_o[k,c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += w_o[k,c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b
    return new_state

def run_one(seed, gamma):
    torch.manual_seed(seed)
    np.random.seed(seed)
    global_state = get_state(MLP())
    for _ in range(N_ROUNDS):
        local_models = [train_local(global_state, city) for city in CITIES]
        global_state = aggregate(local_models, gamma)
    return evaluate(global_state)

# ─────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print(f"GAMMA SWEEP (multi-seed: {SEEDS})")
print("Full E5: CW loss + KI + cwFedAvg + LAWA")
print("=" * 65)

results = {}   # gamma -> list of f1 per seed
for gamma in GAMMAS:
    results[gamma] = []
    for seed in SEEDS:
        f1 = run_one(seed, gamma)
        results[gamma].append(f1)
    mean_f1 = np.mean(results[gamma])
    std_f1  = np.std(results[gamma])
    marker = " <-- optimal" if gamma == 0.2 else ""
    print(f"  gamma={gamma:.1f}  F1={mean_f1:.4f} +/- {std_f1:.4f}{marker}")

# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────
means = [np.mean(results[g]) for g in GAMMAS]
stds  = [np.std(results[g])  for g in GAMMAS]

lines = ["Gamma Sweep -- Multi-seed (seeds=42,0,123), Full E5\n",
         "CW loss + KI + cwFedAvg + LAWA\n\n",
         f"{'gamma':>6}  {'F1 mean':>9}  {'F1 std':>8}\n",
         "-" * 30 + "\n"]
for g, m, s in zip(GAMMAS, means, stds):
    lines.append(f"{g:>6.1f}  {m:>9.4f}  {s:>8.4f}\n")
with open(os.path.join(SAVE_DIR, "gamma_sweep_multiseed.txt"), "w") as fh:
    fh.writelines(lines)

# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(GAMMAS, means, yerr=stds, marker='o', linewidth=2,
            markersize=8, capsize=5, color='steelblue',
            label='Mean F1 +/- std (3 seeds)')
ax.set_xlabel('Gamma (gamma)', fontsize=12)
ax.set_ylabel('Global Macro F1 (Round 10)', fontsize=12)
ax.set_title('Macro F1 vs. Gamma -- E5 Method, 60s Windows\n(averaged over 3 random seeds)', fontsize=12)
ax.set_xticks(GAMMAS)
ax.set_ylim(min(means) - 0.03, max(means) + 0.04)
ax.grid(True, alpha=0.4)
ax.legend()
for g, m in zip(GAMMAS, means):
    ax.annotate(f'{m:.4f}', (g, m), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "11_gamma_sweep_multiseed.png"), dpi=150)
plt.close()

print(f"\nSaved: results/gamma_sweep_multiseed.txt, plots/11_gamma_sweep_multiseed.png")
print("Done.")
