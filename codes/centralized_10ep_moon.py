# centralized_10ep_moon.py
# Two experiments:
#   A) Centralized baseline — 10 epochs (fair compute budget vs FL 10 rounds)
#   B) MOON — model-contrastive FL, temperature=0.5, mu=1, 10 rounds, seed=42

import os, copy, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

WIN_DIR  = r"C:\Users\user\Desktop\electives\VIP\PyTorch\windowed_cache"
SAVE_DIR = r"C:\Users\user\Desktop\electives\VIP\PyTorch\results"
os.makedirs(SAVE_DIR, exist_ok=True)

CITIES      = ['jeddah', 'kaust', 'kz', 'mekkah']
LABEL_MAP   = {'car':0,'walk':1,'bus':2,'scooter':3,'bike':4,
               'motorcycle':5,'jog':6,'train':7}
CLASS_NAMES = ['car','walk','bus','scooter','bike','motorcycle','jog','train']
N_CLASSES   = 8
N_ROUNDS    = 10
BATCH_SIZE  = 32
LR          = 0.01
MOMENTUM    = 0.9
ALL_CLASSES = list(range(N_CLASSES))

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(seed=42):
    city_dfs = {}
    for city in CITIES:
        city_dfs[city] = pd.read_csv(os.path.join(WIN_DIR, f'{city}_60s.csv'))

    feat_cols = [c for c in city_dfs['jeddah'].columns if c != 'label']

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
                                                    random_state=seed, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                    random_state=seed)
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
    n_feat = len(feat_cols)
    return (X_train_d, X_test_d, y_train_d, y_test_d,
            city_classes, X_test_global, y_test_global, n_feat)

# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 256), nn.ReLU(),
            nn.Linear(256, 256),    nn.ReLU(),
        )
        self.fc_out = nn.Linear(256, N_CLASSES)

    def forward(self, x):
        return self.fc_out(self.net(x))

    def get_repr(self, x):
        """256-dim representation before classifier head."""
        return self.net(x)

def get_state(model):    return copy.deepcopy(model.state_dict())
def set_state(model, s): model.load_state_dict(copy.deepcopy(s))

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate(state, X_test_global, y_test_global, X_test_d, y_test_d, n_feat):
    m = MLP(n_feat)
    set_state(m, state)
    m.eval()
    with torch.no_grad():
        preds = m(torch.tensor(X_test_global)).argmax(dim=1).numpy()
    g_acc = float(accuracy_score(y_test_global, preds))
    g_f1  = float(f1_score(y_test_global, preds, average='macro',
                            labels=ALL_CLASSES, zero_division=0))
    city_f1 = {}
    for city in CITIES:
        with torch.no_grad():
            yp = m(torch.tensor(X_test_d[city])).argmax(dim=1).numpy()
        city_f1[city] = float(f1_score(y_test_d[city], yp, average='macro',
                                        labels=ALL_CLASSES, zero_division=0))
    return g_acc, g_f1, city_f1

# ─────────────────────────────────────────────────────────────
# FEDAVG AGGREGATION
# ─────────────────────────────────────────────────────────────

def agg_fedavg(local_states, y_train_d):
    sizes = np.array([len(y_train_d[c]) for c in CITIES], dtype=np.float64)
    w = sizes / sizes.sum()
    new_state = {}
    for key in local_states[0]:
        new_state[key] = sum(w[k] * local_states[k][key].float()
                             for k in range(len(CITIES)))
    return new_state

# ══════════════════════════════════════════════════════════════
# EXPERIMENT A — CENTRALIZED BASELINE (10 EPOCHS)
# ══════════════════════════════════════════════════════════════

def run_centralized_10ep():
    print("\n" + "="*65)
    print("EXPERIMENT A — CENTRALIZED BASELINE (10 EPOCHS)")
    print("="*65)

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    (X_train_d, X_test_d, y_train_d, y_test_d,
     city_classes, X_test_global, y_test_global, n_feat) = load_data(seed)

    # Pool all training data
    X_all = np.vstack([X_train_d[c] for c in CITIES])
    y_all = np.concatenate([y_train_d[c] for c in CITIES]).astype(np.int64)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_all), torch.tensor(y_all)),
        batch_size=BATCH_SIZE, shuffle=True)

    # ── Unweighted ──
    torch.manual_seed(seed); np.random.seed(seed)
    model_uw  = MLP(n_feat)
    opt_uw    = optim.SGD(model_uw.parameters(), lr=LR, momentum=MOMENTUM)
    crit_uw   = nn.CrossEntropyLoss()
    for epoch in range(1, 11):
        model_uw.train()
        for Xb, yb in loader:
            opt_uw.zero_grad()
            crit_uw(model_uw(Xb), yb).backward()
            opt_uw.step()
    uw_acc, uw_f1, uw_city = evaluate(get_state(model_uw), X_test_global,
                                       y_test_global, X_test_d, y_test_d, n_feat)
    print(f"  Unweighted (10 ep): Acc={uw_acc:.4f}  MacroF1={uw_f1:.4f}")

    # ── Class-weighted ──
    torch.manual_seed(seed); np.random.seed(seed)
    n_tot    = len(y_all)
    weights  = torch.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        n_c = (y_all == c).sum()
        if n_c > 0:
            weights[c] = n_tot / (N_CLASSES * n_c)
    crit_cw  = nn.CrossEntropyLoss(weight=weights)
    model_cw = MLP(n_feat)
    opt_cw   = optim.SGD(model_cw.parameters(), lr=LR, momentum=MOMENTUM)
    for epoch in range(1, 11):
        model_cw.train()
        for Xb, yb in loader:
            opt_cw.zero_grad()
            crit_cw(model_cw(Xb), yb).backward()
            opt_cw.step()
    cw_acc, cw_f1, cw_city = evaluate(get_state(model_cw), X_test_global,
                                       y_test_global, X_test_d, y_test_d, n_feat)
    print(f"  Class-weighted (10 ep): Acc={cw_acc:.4f}  MacroF1={cw_f1:.4f}")

    lines = [
        "Centralized Baseline — 10 Epochs (fair budget vs FL 10 rounds)\n",
        f"\n{'Method':<35} {'Accuracy':>10} {'Macro F1':>10}\n",
        "-"*58 + "\n",
        f"{'Centralized (unweighted)':<35} {uw_acc:>10.4f} {uw_f1:>10.4f}\n",
        f"{'Centralized (class-weighted)':<35} {cw_acc:>10.4f} {cw_f1:>10.4f}\n",
        "\nPer-city Macro F1:\n",
        f"  {'City':<12} {'Unweighted':>12} {'CW-weighted':>12}\n",
        "  " + "-"*38 + "\n",
    ]
    for city in CITIES:
        lines.append(f"  {city:<12} {uw_city[city]:>12.4f} {cw_city[city]:>12.4f}\n")

    with open(os.path.join(SAVE_DIR, "centralized_10ep.txt"), "w") as fh:
        fh.writelines(lines)
    print(f"  Saved: centralized_10ep.txt")
    return uw_acc, uw_f1, cw_acc, cw_f1

# ══════════════════════════════════════════════════════════════
# EXPERIMENT B — MOON
# ══════════════════════════════════════════════════════════════
# Li et al., "Model-Contrastive Federated Learning", CVPR 2021
# L_total = L_sup + mu * L_con
# L_con   = -log[ exp(sim(z, z_glob)/tau) /
#                 (exp(sim(z, z_glob)/tau) + exp(sim(z, z_prev)/tau)) ]
# z       = repr from current local model (detach positives/negatives)
# z_glob  = repr from global model (frozen)
# z_prev  = repr from previous local model (frozen, = z_glob on round 1)

def moon_contrastive_loss(z, z_glob, z_prev, tau=0.5):
    """Per-sample InfoNCE with one positive (global) and one negative (prev)."""
    z      = F.normalize(z,      dim=1)
    z_glob = F.normalize(z_glob, dim=1)
    z_prev = F.normalize(z_prev, dim=1)

    sim_pos = (z * z_glob).sum(dim=1) / tau   # (B,)
    sim_neg = (z * z_prev).sum(dim=1) / tau   # (B,)

    # log-softmax over [pos, neg]
    logits = torch.stack([sim_pos, sim_neg], dim=1)  # (B, 2)
    labels = torch.zeros(z.size(0), dtype=torch.long)  # positive is index 0
    return F.cross_entropy(logits, labels)


def run_moon():
    print("\n" + "="*65)
    print("EXPERIMENT B — MOON (tau=0.5, mu=1, 10 rounds, seed=42)")
    print("="*65)

    seed = 42
    tau  = 0.5
    mu   = 1.0

    torch.manual_seed(seed); np.random.seed(seed)
    (X_train_d, X_test_d, y_train_d, y_test_d,
     city_classes, X_test_global, y_test_global, n_feat) = load_data(seed)

    global_state = get_state(MLP(n_feat))
    # Previous local model state per city — initialised to global on round 1
    prev_states  = {city: copy.deepcopy(global_state) for city in CITIES}

    per_class_f1_last = None

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        local_states = []
        new_prev     = {}

        for city in CITIES:
            # Build frozen reference models
            global_model = MLP(n_feat)
            set_state(global_model, global_state)
            global_model.eval()

            prev_model = MLP(n_feat)
            set_state(prev_model, prev_states[city])
            prev_model.eval()

            # Local model to train
            model = MLP(n_feat)
            set_state(model, global_state)
            model.train()

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
            loader    = DataLoader(
                TensorDataset(torch.tensor(X_train_d[city]),
                              torch.tensor(y_train_d[city])),
                batch_size=BATCH_SIZE, shuffle=True)

            for Xb, yb in loader:
                optimizer.zero_grad()

                # Supervised loss
                logits = model(Xb)
                l_sup  = criterion(logits, yb)

                # Contrastive loss
                with torch.no_grad():
                    z_glob = global_model.get_repr(Xb)
                    z_prev = prev_model.get_repr(Xb)
                z = model.get_repr(Xb)
                l_con = moon_contrastive_loss(z, z_glob, z_prev, tau=tau)

                (l_sup + mu * l_con).backward()
                optimizer.step()

            new_prev[city] = get_state(model)
            local_states.append(get_state(model))

        prev_states  = new_prev
        global_state = agg_fedavg(local_states, y_train_d)

        g_acc, g_f1, city_f1 = evaluate(global_state, X_test_global,
                                          y_test_global, X_test_d, y_test_d, n_feat)
        print(f"  Round {rnd:2d}/10  Acc {g_acc:.4f}  MacroF1 {g_f1:.4f}"
              f"  ({time.time()-t0:.1f}s)")

        if rnd == N_ROUNDS:
            # Per-class F1 for final round
            m = MLP(n_feat)
            set_state(m, global_state)
            m.eval()
            with torch.no_grad():
                preds = m(torch.tensor(X_test_global)).argmax(dim=1).numpy()
            per_class_f1_last = f1_score(y_test_global, preds, average=None,
                                          labels=ALL_CLASSES, zero_division=0)

    lines = [
        "MOON — tau=0.5, mu=1, 60s Windows, Round 10, Seed 42\n",
        f"\nGlobal Accuracy: {g_acc:.4f}\n",
        f"Global Macro F1: {g_f1:.4f}\n",
        "\nPer-city Macro F1 (global model):\n",
    ]
    for city in CITIES:
        lines.append(f"  {city:<10}: {city_f1[city]:.4f}\n")
    lines += ["\nPer-class F1 (global model):\n"]
    for i, cls in enumerate(CLASS_NAMES):
        lines.append(f"  {cls:<14}: {per_class_f1_last[i]:.4f}\n")

    with open(os.path.join(SAVE_DIR, "moon.txt"), "w") as fh:
        fh.writelines(lines)

    print(f"  Global model:  Acc={g_acc:.4f}  MacroF1={g_f1:.4f}")
    print(f"  Per-city: " + "  ".join(f"{c}={city_f1[c]:.4f}" for c in CITIES))
    print(f"  Per-class F1: " + "  ".join(
          f"{CLASS_NAMES[i]}={per_class_f1_last[i]:.3f}"
          for i in range(len(CLASS_NAMES))))
    print(f"  Saved: moon.txt")
    return g_acc, g_f1, city_f1

# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t_start = time.time()
    print("="*65)
    print("CENTRALIZED (10-EP) + MOON")
    print("="*65)

    uw_acc, uw_f1, cw_acc, cw_f1 = run_centralized_10ep()
    moon_acc, moon_f1, moon_city  = run_moon()

    total_min = (time.time() - t_start) / 60
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"  Centralized unweighted (10ep):    Acc={uw_acc:.4f}  MacroF1={uw_f1:.4f}")
    print(f"  Centralized class-weighted (10ep): Acc={cw_acc:.4f}  MacroF1={cw_f1:.4f}")
    print(f"  MOON (tau=0.5, mu=1):              Acc={moon_acc:.4f}  MacroF1={moon_f1:.4f}")
    print(f"  Done in {total_min:.1f} min")
    print("="*65)
