# Ablation study: 60-second windows, PyTorch FedAvg
# Adds one component at a time across 5 experiments:
#   E1: FedAvg baseline
#   E2: + class-weighted loss
#   E3: + knowledge inheritance
#   E4: + cwFedAvg aggregation (class-wise output layer)
#   E5: + LAWA blend (gamma=0.2)
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
MOMENTUM    = 0.9
GAMMA       = 0.2
ALL_CLASSES = list(range(N_CLASSES))

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# LOAD 60s WINDOWED CACHE
# -----------------------------------------------------------------------------
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

# Split + encode
X_train_d, X_test_d, y_train_d, y_test_d = {}, {}, {}, {}
city_classes = {}  # set of int class ids present in each city's training split

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

    present_names = [CLASS_NAMES[c] for c in sorted(city_classes[city])]
    print(f"  {city:8s}  train {len(ytr):4d}  test {len(yte):3d}  classes={present_names}")

# Global StandardScaler
scaler = StandardScaler()
scaler.fit(np.vstack([X_train_d[c] for c in CITIES]))
for city in CITIES:
    X_train_d[city] = scaler.transform(X_train_d[city]).astype(np.float32)
    X_test_d[city]  = scaler.transform(X_test_d[city]).astype(np.float32)

X_test_global = np.vstack([X_test_d[c]  for c in CITIES])
y_test_global = np.concatenate([y_test_d[c] for c in CITIES])
print(f"\n  Global test: {len(y_test_global)} samples")

# -----------------------------------------------------------------------------
# MODEL  (named fc_out needed for knowledge inheritance)
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

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def get_state(model):    return copy.deepcopy(model.state_dict())
def set_state(model, s): model.load_state_dict(copy.deepcopy(s))

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
# LOCAL TRAINING VARIANTS
# -----------------------------------------------------------------------------
def train_local(global_state, city, use_cw_loss=False, use_ki=False):
    """
    use_cw_loss : class-weighted CrossEntropyLoss
    use_ki      : knowledge inheritance for missing classes
    """
    model = MLP()
    set_state(model, global_state)
    model.train()

    if use_cw_loss:
        y_arr  = y_train_d[city]
        n_tot  = len(y_arr)
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

# -----------------------------------------------------------------------------
# AGGREGATION VARIANTS
# -----------------------------------------------------------------------------
def agg_fedavg(local_models):
    """Standard FedAvg: all layers weighted by sample count."""
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w = sizes / sizes.sum()
    new_state = {}
    for key in local_models[0].state_dict():
        new_state[key] = sum(
            w[k] * local_models[k].state_dict()[key].float()
            for k in range(len(CITIES))
        )
    return new_state


def agg_cw(local_models):
    """cwFedAvg: hidden layers by sample count, output layer class-by-class."""
    n     = len(local_models)
    cc    = np.array([class_count_vec(y_train_d[c]) for c in CITIES])  # (n, N_CLASSES)
    sizes = cc.sum(axis=1)          # (n,)
    total = sizes.sum()
    w_scalar = sizes / total        # (n,) for hidden layers

    ct   = cc.sum(axis=0)           # (N_CLASSES,) total per class
    w_cw = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)  # (n, N_CLASSES)

    new_state = {}
    for key in local_models[0].state_dict():
        if 'fc_out' in key:
            continue
        new_state[key] = sum(
            w_scalar[k] * local_models[k].state_dict()[key].float()
            for k in range(n)
        )

    # Output layer: per-class weighted
    out_w = torch.zeros_like(local_models[0].state_dict()['fc_out.weight'])
    out_b = torch.zeros_like(local_models[0].state_dict()['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += w_cw[k, c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += w_cw[k, c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b
    return new_state


def compute_lawa_weights(local_models):
    """L_min/L_max ratio per client, normalised."""
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


def agg_cw_lawa(local_models, gamma=GAMMA):
    """cwFedAvg + LAWA blend (gamma=0.2)."""
    n     = len(local_models)
    cc    = np.array([class_count_vec(y_train_d[c]) for c in CITIES])  # (n, N_CLASSES)
    sizes = cc.sum(axis=1)
    total = sizes.sum()
    w_scalar = sizes / total
    ct   = cc.sum(axis=0)
    w_cw = np.where(ct > 0, cc / np.maximum(ct, 1e-12), 1.0 / n)

    lawa_w = compute_lawa_weights(local_models)  # (n,)

    # Blended weights
    bl_s  = gamma * lawa_w + (1 - gamma) * w_scalar     # (n,)
    bl_s /= bl_s.sum()

    bl_cw = gamma * lawa_w[:, None] + (1 - gamma) * w_cw  # (n, N_CLASSES)
    for c in range(N_CLASSES):
        s = bl_cw[:, c].sum()
        if s > 0:
            bl_cw[:, c] /= s

    new_state = {}
    for key in local_models[0].state_dict():
        if 'fc_out' in key:
            continue
        new_state[key] = sum(
            bl_s[k] * local_models[k].state_dict()[key].float()
            for k in range(n)
        )

    out_w = torch.zeros_like(local_models[0].state_dict()['fc_out.weight'])
    out_b = torch.zeros_like(local_models[0].state_dict()['fc_out.bias'])
    for c in range(N_CLASSES):
        for k in range(n):
            out_w[c] += bl_cw[k, c] * local_models[k].state_dict()['fc_out.weight'][c].float()
            out_b[c] += bl_cw[k, c] * local_models[k].state_dict()['fc_out.bias'][c].float()
    new_state['fc_out.weight'] = out_w
    new_state['fc_out.bias']   = out_b
    return new_state

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
def run_experiment(label, use_cw_loss, use_ki, agg_fn, save_name):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")

    global_state = get_state(MLP())
    rows = []
    t0_exp = time.time()

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        local_models = [
            train_local(global_state, city, use_cw_loss=use_cw_loss, use_ki=use_ki)
            for city in CITIES
        ]
        global_state = agg_fn(local_models)
        g_acc, g_f1, pc_f1, city_m = evaluate(global_state)
        print(f"  Round {rnd:2d}/10  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}  ({time.time()-t0:.1f}s)")

        row = {'experiment': label, 'round': rnd,
               'global_acc': round(g_acc, 6), 'global_macro_f1': round(g_f1, 6)}
        for city in CITIES:
            row[f'{city}_macro_f1'] = round(city_m[city], 6)
        for i, cls in enumerate(CLASS_NAMES):
            row[f'f1_{cls}'] = round(float(pc_f1[i]), 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    out = os.path.join(SAVE_DIR, save_name)
    df.to_csv(out, index=False)

    last = df[df['round'] == N_ROUNDS].iloc[0]
    total_min = (time.time() - t0_exp) / 60
    print(f"  Done in {total_min:.1f} min  ->  {out}")
    print(f"  Round 10 -> Acc {last['global_acc']:.4f}  MacroF1 {last['global_macro_f1']:.4f}")
    print(f"  Per-city MacroF1: " +
          "  ".join(f"{c}={last[f'{c}_macro_f1']:.4f}" for c in CITIES))
    print(f"  Per-class F1:")
    for cls in CLASS_NAMES:
        print(f"    {cls:12s}  {last[f'f1_{cls}']:.4f}")
    return df

# -----------------------------------------------------------------------------
# RUN ALL ABLATION EXPERIMENTS
# -----------------------------------------------------------------------------
print(f"\n{'='*65}")
print("ABLATION STUDY  (60s windows, 10 rounds each)")
print(f"{'='*65}")

experiments = [
    dict(label="E1: FedAvg baseline",
         use_cw_loss=False, use_ki=False, agg_fn=agg_fedavg,
         save_name="ablation_e1_baseline.csv"),

    dict(label="E2: + class-weighted loss",
         use_cw_loss=True,  use_ki=False, agg_fn=agg_fedavg,
         save_name="ablation_e2_cwloss.csv"),

    dict(label="E3: + knowledge inheritance",
         use_cw_loss=True,  use_ki=True,  agg_fn=agg_fedavg,
         save_name="ablation_e3_ki.csv"),

    dict(label="E4: + cwFedAvg aggregation",
         use_cw_loss=True,  use_ki=True,  agg_fn=agg_cw,
         save_name="ablation_e4_cwagg.csv"),

    dict(label="E5: + LAWA blend (gamma=0.2)",
         use_cw_loss=True,  use_ki=True,  agg_fn=agg_cw_lawa,
         save_name="ablation_e5_lawa.csv"),
]

all_dfs = []
for exp in experiments:
    df = run_experiment(**exp)
    all_dfs.append(df)

# -----------------------------------------------------------------------------
# COMBINED SAVE + SUMMARY TABLE
# -----------------------------------------------------------------------------
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(os.path.join(SAVE_DIR, "ablation_60s_all.csv"), index=False)

last_rows = combined[combined['round'] == N_ROUNDS].copy()

print(f"\n{'='*65}")
print("ABLATION SUMMARY  (Round 10, 60s windows)")
print(f"{'='*65}")
print(f"\n  {'Experiment':<38} {'Acc':>7}  {'MacroF1':>8}")
print(f"  {'-'*58}")
for _, row in last_rows.iterrows():
    print(f"  {row['experiment']:<38} {row['global_acc']:>7.4f}  {row['global_macro_f1']:>8.4f}")

print(f"\n  Per-class F1 at Round 10:")
header = f"  {'Experiment':<38} " + "  ".join(f"{cls[:5]:>6}" for cls in CLASS_NAMES)
print(header)
print(f"  {'-'*90}")
for _, row in last_rows.iterrows():
    vals = "  ".join(f"{row[f'f1_{cls}']:>6.3f}" for cls in CLASS_NAMES)
    print(f"  {row['experiment']:<38} {vals}")

print(f"\nAll results saved to: {SAVE_DIR}")
print("Done.")
