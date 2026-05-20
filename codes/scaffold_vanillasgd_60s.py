# SCAFFOLD with vanilla SGD (momentum=0) on 60s windowed data
# Same MLP (256-256), lr=0.01, batch_size=32, 10 rounds as all prior experiments.
# Momentum is set to 0 because the SCAFFOLD control variate update formula:
#   c_i_new = c_i - c + (w_global - w_local) / (K * lr)
# is derived assuming plain SGD. With momentum the correction diverges.
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

WIN_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"
SAVE_DIR = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(SAVE_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.0   # vanilla SGD — required for correct SCAFFOLD gradient correction
ALL_CLASSES = list(range(N_CLASSES))

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# LOAD 60s WINDOWED CACHE
# -----------------------------------------------------------------------------
print("=" * 65)
print("SCAFFOLD — vanilla SGD (momentum=0), 60s windows")
print("=" * 65)

city_dfs = {}
for city in CITIES:
    city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))
    print(f"  {city:8s}: {len(city_dfs[city])} windows")

feat_cols  = [c for c in city_dfs['jeddah'].columns if c != 'label']
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
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42)
    X_train_d[city] = Xtr
    X_test_d[city]  = Xte
    y_train_d[city] = ytr.astype(np.int64)
    y_test_d[city]  = yte.astype(np.int64)
    city_classes[city] = set(int(c) for c in np.unique(ytr))
    present = [CLASS_NAMES[c] for c in sorted(city_classes[city])]
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}  classes={present}")

scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"\n  Global test: {len(y_test_global)} samples")

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
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

def get_state(model):    return copy.deepcopy(model.state_dict())
def set_state(model, s): model.load_state_dict(copy.deepcopy(s))
def zero_state():
    return {k: torch.zeros_like(v) for k, v in MLP().state_dict().items()}

# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate(global_state):
    m = MLP(); set_state(m, global_state); m.eval()
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

# -----------------------------------------------------------------------------
# FEDAVG AGGREGATION (weighted by sample count)
# -----------------------------------------------------------------------------
def fedavg_agg(local_states):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w = sizes / sizes.sum()
    new_state = {}
    for key in local_states[0]:
        new_state[key] = sum(w[k] * local_states[k][key].float()
                             for k in range(len(CITIES)))
    return new_state

# -----------------------------------------------------------------------------
# SCAFFOLD LOCAL TRAINING (vanilla SGD)
# -----------------------------------------------------------------------------
def train_scaffold(global_state, city, c_global, c_local):
    model = MLP(); set_state(model, global_state); model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train_d[city]),
                      torch.tensor(y_train_d[city])),
        batch_size=BATCH_SIZE, shuffle=True)

    K = 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        # Control-variate correction: grad += c_global - c_local
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad.add_(c_global[name] - c_local[name])
        optimizer.step()
        K += 1

    new_state = get_state(model)

    # c_i_new = c_i - c + (w_global - w_local) / (K * lr)
    new_c_local, delta_c = {}, {}
    for key in global_state:
        correction       = (global_state[key].float() - new_state[key].float()) / (K * LR)
        new_c_local[key] = c_local[key] + correction - c_global[key]
        delta_c[key]     = new_c_local[key] - c_local[key]

    return new_state, new_c_local, delta_c

# -----------------------------------------------------------------------------
# SCAFFOLD TRAINING LOOP
# -----------------------------------------------------------------------------
print(f"\n{'='*65}")
print("  Running SCAFFOLD (10 rounds)")
print(f"{'='*65}")

c_global = zero_state()
c_locals = {city: zero_state() for city in CITIES}
global_state = get_state(MLP())
rows = []
t0_exp = time.time()

for rnd in range(1, N_ROUNDS + 1):
    t0 = time.time()
    local_states, delta_cs = [], []

    for city in CITIES:
        new_state, new_c_local, delta_c = train_scaffold(
            global_state, city, c_global, c_locals[city])
        local_states.append(new_state)
        delta_cs.append(delta_c)
        c_locals[city] = new_c_local

    # Aggregate model
    global_state = fedavg_agg(local_states)

    # Update server control variate: c += (1/n) * sum(delta_c_i)
    n = len(CITIES)
    for key in c_global:
        c_global[key] = c_global[key] + sum(dc[key] for dc in delta_cs) / n

    g_acc, g_f1, pc_f1, city_m = evaluate(global_state)
    print(f"  Round {rnd:2d}/10  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

    row = {'experiment': 'SCAFFOLD (momentum=0)', 'round': rnd,
           'global_acc': round(g_acc, 6), 'global_macro_f1': round(g_f1, 6)}
    for city in CITIES:
        row[f'{city}_macro_f1'] = round(city_m[city], 6)
    for i, cls in enumerate(CLASS_NAMES):
        row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
    rows.append(row)

df = pd.DataFrame(rows)
out = os.path.join(SAVE_DIR, 'scaffold_vanillasgd_60s.csv')
df.to_csv(out, index=False)

last = df[df['round'] == N_ROUNDS].iloc[0]
total_min = (time.time() - t0_exp) / 60

print(f"\n  Done in {total_min:.1f} min  ->  {out}")
print(f"\n--- Round 10 Results ---")
print(f"  Global Accuracy : {last['global_acc']:.4f}")
print(f"  Global Macro F1 : {last['global_macro_f1']:.4f}")
print(f"  Per-city Macro F1:")
for city in CITIES:
    print(f"    {city:8s}  {last[f'{city}_macro_f1']:.4f}")
print(f"  Per-class F1:")
for cls in CLASS_NAMES:
    print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")

# -----------------------------------------------------------------------------
# COMPARISON TABLE
# -----------------------------------------------------------------------------
ref = [
    ('FedAvg baseline',      0.7837, 0.5243, 0.2989, 0.5412, 0.1554, 0.2324,
     0.643, 0.943, 0.716, 0.818, 0.675, 0.400, 0.000, 0.000),
    ('FedProx (mu=0.01)',    0.7827, 0.5220, 0.2989, 0.5384, 0.1554, 0.2351,
     0.643, 0.944, 0.719, 0.812, 0.658, 0.400, 0.000, 0.000),
    ('SCAFFOLD (momentum=0.9 — collapsed)', 0.1891, 0.0398, 0.0700, 0.0131, 0.1203, 0.0538,
     0.318, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000),
]

print(f"\n{'='*65}")
print("FULL COMPARISON  (Round 10, 60s windows)")
print(f"{'='*65}")
print(f"\n  {'Experiment':<40} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*60}")
for name, acc, f1, *_ in ref:
    print(f"  {name:<40} {acc:>7.4f}  {f1:>8.4f}")
print(f"  {'SCAFFOLD (momentum=0)':<40} {last['global_acc']:>7.4f}  {last['global_macro_f1']:>8.4f}")

print(f"\n  Per-city Macro F1:")
print(f"  {'Experiment':<40} {'jeddah':>8}  {'kaust':>8}  {'kz':>6}  {'mekkah':>8}")
print(f"  {'-'*74}")
for name, acc, f1, jed, kau, kz, mek, *_ in ref:
    print(f"  {name:<40} {jed:>8.4f}  {kau:>8.4f}  {kz:>6.4f}  {mek:>8.4f}")
print(f"  {'SCAFFOLD (momentum=0)':<40} "
      f"{last['jeddah_macro_f1']:>8.4f}  {last['kaust_macro_f1']:>8.4f}  "
      f"{last['kz_macro_f1']:>6.4f}  {last['mekkah_macro_f1']:>8.4f}")

print(f"\n  Per-class F1:")
print(f"  {'Experiment':<40} " + "  ".join(f"{cls[:5]:>6}" for cls in CLASS_NAMES))
print(f"  {'-'*95}")
for name, acc, f1, jed, kau, kz, mek, *cls_f1s in ref:
    vals = "  ".join(f"{v:>6.3f}" for v in cls_f1s)
    print(f"  {name:<40} {vals}")
scaffold_vals = "  ".join(f"{last[f'f1_{cls}']:>6.3f}" for cls in CLASS_NAMES)
print(f"  {'SCAFFOLD (momentum=0)':<40} {scaffold_vals}")

print(f"\nSaved -> {out}")
print("Done.")
