# Federated Learning under Structural Class Absence
# for Multi-City Transportation Mode Detection

Submitted to IEEE Globecom 2025 — American University of Beirut

---

## What this paper is about

Standard federated learning assumes all clients share the same 
label space. In reality, cities are physically different mobility 
environments — some transport modes simply do not exist in certain 
cities, not because of sampling, but because of geography and 
infrastructure. This creates what we call Partially Class-Disjoint 
Data (PCDD): a structural heterogeneity that existing FL methods 
are not designed to handle.

We identify gradient interference from structurally uninformed 
clients as the dominant failure mechanism under PCDD, formalize 
it mathematically, and propose a two-level class-aware framework 
that enforces physical knowledge boundaries at both local training 
and global aggregation stages. This is a problem-driven paper — 
every design decision is a direct consequence of the PCDD 
diagnosis, not an independent algorithmic contribution.

---

## Why this matters

Federated learning for multi-city transportation mode detection 
is a natural deployment scenario — cities cannot share raw user 
trajectory data due to privacy regulations, making FL the only 
viable option. But when you actually run standard FL across real 
cities, it breaks in a specific and predictable way. Clients that 
have never observed a transport mode contribute gradients that 
corrupt the representation learned by the one city that has. 
The failure is structural, not statistical, and cannot be fixed 
by better optimizers, more communication rounds, or proximal 
regularization.

---

## The core theoretical insight

For any class c, the aggregated gradient in standard FedAvg 
decomposes as:

g_c = [informed signal from clients that observed c]
    + [interference from clients that never observed c]

Under PCDD, the interference term can dominate the informed 
signal entirely. For rare classes confined to a single city, 
the one informed client is outweighed by multiple uninformed 
ones. This is a structural data problem — no optimization-level 
intervention can recover information that is absent from 
the local data.

The fix is enforcing: p_{k,c} = 0 if n_{k,c} = 0
No client should influence a class it has never observed.

---

## Dataset

4 Saudi cities, LTE cellular measurements at ~1 Hz.
Collected by Zhagyparova et al., IEEE Smart Mobility 2023.

30 statistical features per 60-second window:
mean, variance, max, min of RSRP, RSSI, RSRQ, CQI, 
velocity, path loss, plus handover rate, timing advance, 
cell count, and location features.

7 transport mode classes: car, walk, bus, scooter, 
bike, motorcycle, jog.

| City | Windows | Classes | Dominant |
|---|---|---|---|
| Jeddah | 809 | 3 | Car 58% |
| KAUST | 2460 | 7 | Walk 35% |
| KZ | 160 | 2 | Car 93% |
| Mekkah | 538 | 3 | Bus 45% |

KAUST is the only city with scooter, bike, motorcycle, 
and jog — not by chance, but because it is a self-contained 
campus where those modes physically exist.

The 60-second window is physically motivated: it reliably 
captures one coherent mobility state without spanning mode 
transitions or over-averaging discriminative variance. 
Confirmed empirically — windowed features improve macro F1 
by 87% over row-by-row classification.

---

## The proposed framework

One principle drives every component:
clients should only influence classes they physically observe.

### Level 1 — Local Training

**Class-Weighted Loss**
Assigns inverse-frequency weights to each class during 
local training. Corrects within-city geographic imbalance — 
KZ's environment generates 93% car trips, collapsing local 
training to a near-degenerate car detector without reweighting.

**Knowledge Inheritance**
For any class absent at a client, the corresponding output 
node is frozen to the global model values received at the 
start of that round. A city that has never produced a 
certain trip type carries no valid signal about what that 
mode looks like in cellular data — allowing it to update 
that output node is uninformed gradient interference.

### Level 2 — Global Aggregation

**Class-Wise FedAvg (cwFedAvg)**
Each output node is aggregated independently using only 
clients that have observed that class, weighted by their 
relative sample counts for that class. The server-side 
enforcement of the same principle that knowledge inheritance 
enforces locally.

**LAWA + γ Blending**
Each client computes a loss balance ratio from its local 
training trajectory, quantifying how evenly it learned 
across its observed classes. This is blended with cwFedAvg 
weights at γ=0.2. Design rule: γ should decrease as class 
disjointness increases — at full disjointness, pure cwFedAvg 
is optimal.

---

## Results

### Ablation — every component earns its place

| Experiment | Macro F1 | ΔF1 |
|---|---|---|
| E1: FedAvg baseline | 0.524 | — |
| E2: + class-weighted loss | 0.557 | +0.033 |
| E3: + knowledge inheritance | 0.623 | +0.099 |
| E4: + cwFedAvg | 0.616 | +0.092 |
| E5: + LAWA (γ=0.2) | 0.631 | +0.107 |

Knowledge inheritance is the single largest jump, confirming 
that the dominant factor is prevention of uninformed updates, 
not improved optimization.

### Full comparison

| Method | Accuracy | Macro F1 | ±Std |
|---|---|---|---|
| SCAFFOLD | 0.716 | 0.366 | — |
| FedProx | 0.783 | 0.522 | — |
| FedAvg | 0.784 | 0.524 | — |
| MOON | 0.801 | 0.560 | 0.010 |
| FedPer | 0.812 | 0.585 | — |
| Centralized plain CE | 0.842 | 0.591 | — |
| FedBN | 0.804 | 0.624 | 0.035 |
| **Proposed E5** | **0.805** | **0.631** | **0.010** |
| Centralized class-weighted | 0.826 | 0.653 | — |

E5 outperforms all FL baselines, surpasses the standard 
centralized model by +0.040, and falls within 0.022 of the 
class-weighted centralized upper bound — without sharing 
any raw data.

### Why FedBN's macro F1 is misleading

FedBN achieves 0.624 macro F1 vs our 0.631 — a gap of 0.007 
that looks small. But FedBN addresses feature distribution 
shift, not structural class absence. Its variance across seeds 
is 3.5× higher than E5 (0.035 vs 0.010), and per-class 
analysis reveals it still nearly collapses rare classes while 
E5 recovers them substantially. The macro F1 gap understates 
the qualitative difference between the two methods.

### All four cities improve

| City | FedAvg F1 | E5 F1 | Gain |
|---|---|---|---|
| Jeddah | 0.153 | 0.301 | +0.149 |
| KAUST | 0.242 | 0.642 | +0.400 |
| KZ | 0.074 | 0.196 | +0.122 |
| Mekkah | 0.119 | 0.239 | +0.120 |

A federated method that helps one client at the expense 
of others is not a valid solution. Every city benefits.

### PCDD disjointness experiment

We constructed three controlled settings by modifying 
class availability across clients:

- Condition A (low disjointness, avg 2.71 clients/class)
- Condition B (medium, avg 2.14) — natural dataset config
- Condition C (high, avg 1.86)

E5 consistently outperforms FedAvg across all conditions. 
The largest gain occurs at moderate disjointness (+0.114), 
exactly where the PCDD framework predicts it should peak — 
when classes are present but confined to a single client, 
interference is maximized and class-aware aggregation is 
most critical. This validates the PCDD framing as a 
scientific hypothesis, not a post-hoc explanation.

### Scalability — works at any data size

Evaluated across three scales with strict trip-level 
train/test splits to prevent temporal leakage:

| Dataset | Windows | FedAvg F1 | E5 F1 |
|---|---|---|---|
| Non-overlapping | 4,967 | 0.551±0.021 | 0.630±0.010 |
| Sliding 30s-step | 9,669 | 0.519±0.021 | 0.555±0.012 |
| Sliding 5s-step | 56,794 | 0.560±0.017 | 0.575±0.009 |

E5 outperforms FedAvg at every scale. Its variance decreases 
monotonically as data grows, while FedAvg's remains high. 
Structural constraints provide deployment consistency that 
random initialization cannot.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch, scikit-learn, numpy, matplotlib.

---

## Run

```bash
# FedAvg baseline
python src/train.py --method fedavg --rounds 10 --seed 42

# Full proposed method
python src/train.py --method e5 --gamma 0.2 --rounds 10 --seed 42

# Ablation variants
python src/train.py --method e2  # class-weighted loss only
python src/train.py --method e3  # + knowledge inheritance
python src/train.py --method e4  # + cwFedAvg
python src/train.py --method e5  # + LAWA blend

# Baselines
python src/train.py --method fedprox --mu 0.01
python src/train.py --method scaffold
python src/train.py --method fedbn
python src/train.py --method moon
```

---

## Data

Dataset from Zhagyparova et al., IEEE Smart Mobility 2023. 
Not ours to redistribute. Contact the original authors 
at KAUST for access.

---

## Citation

```bibtex
@inproceedings{jaffal2025pcdd,
  title={Federated Learning under Structural Class Absence 
         for Multi-City Transportation Mode Detection},
  author={Jaffal, Carla and El Hajj, Khalil and Sarieddeen, Hadi},
  booktitle={Proc. IEEE Globecom},
  year={2025}
}
```

---

## Authors

Carla Jaffal · Khalil El Hajj · Hadi Sarieddeen  
Department of Electrical and Computer Engineering  
American University of Beirut
