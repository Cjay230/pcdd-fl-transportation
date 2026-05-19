# Federated Learning under Structural Class Absence for Multi-City Transportation Mode Detection

Paper submitted to IEEE Globecom 2025.

Cities don't just have different data distributions — they have 
physically different mobility ecosystems. KAUST is a campus where 
people jog and cycle. Jeddah is a highway city. KZ is almost 
entirely car trips. When you run standard federated learning across 
these cities, clients that have never seen a jogging trip still 
contribute gradients that overwrite the one city that has. The result 
is that FedAvg collapses jog classification to F1 = 0.000.

This repo contains the code for our paper, which identifies this as 
a Partially Class-Disjoint Data (PCDD) problem and proposes a 
framework that fixes it.

## The problem in one equation

When aggregating gradients for class c:

g_c = (informed signal from cities that have seen c) + (interference from cities that haven't)

For jog: KAUST contributes ~49.5% of the weight. 
The three other cities contribute ~50.5%. 
Interference wins. Jog dies.

## What we do about it

Two levels, one principle: a city should only influence 
the representation of classes it physically produces.

**Locally:**
- Knowledge Inheritance — freeze output nodes for classes 
  a client has never seen. Don't let Jeddah update the jog node.
- Class-Weighted Loss — correct within-city imbalance. 
  KZ is 93% car; without reweighting the local model 
  becomes a car detector.

**Globally:**
- cwFedAvg — aggregate each class independently, 
  using only cities that have observed it.
- LAWA — weight clients by how balanced their local 
  training was, blended with cwFedAvg at γ=0.2.

## Results

| Method | Macro F1 | Jog F1 |
|---|---|---|
| FedAvg | 0.524 | 0.000 |
| FedProx | 0.522 | ~0.000 |
| SCAFFOLD | 0.366 | ~0.000 |
| FedBN | 0.624 | 0.333 |
| MOON | 0.560 | — |
| Centralized (plain) | 0.591 | — |
| **Ours (E5)** | **0.631** | **0.567** |
| Centralized (class-weighted) | 0.653 | — |

FedBN gets close on macro F1 but still collapses jog. 
Our method's variance across seeds is 3.5x lower than FedBN 
(0.010 vs 0.035) — it's not getting lucky, it's structurally stable.

## Dataset

4 Saudi cities, LTE cellular measurements at ~1Hz.
Features: RSRP, RSSI, RSRQ, CQI, velocity, path loss, 
handover rate, timing advance, cell count — 
30 statistical features per 60-second window.

7 classes: car, walk, bus, scooter, bike, motorcycle, jog.
Only KAUST has the last 4.

Data from Zhagyparova et al., IEEE Smart Mobility 2023. 
Not ours to redistribute — contact the original authors.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.8+, PyTorch, scikit-learn, numpy, matplotlib.

## Run

```bash
# baseline
python src/fedavg.py

# full proposed method
python src/train.py --method e5 --gamma 0.2 --rounds 10 --seed 42

# ablation
python src/train.py --method e3  # knowledge inheritance only
python src/train.py --method e4  # + cwFedAvg
```
## Ablation

| Experiment | Macro F1 | ΔF1 |
|---|---|---|
| E1: FedAvg baseline | 0.524 | — |
| E2: + class-weighted loss | 0.557 | +0.033 |
| E3: + knowledge inheritance | 0.623 | +0.099 |
| E4: + cwFedAvg | 0.616 | +0.092 |
| E5: + LAWA (γ=0.2) | 0.631 | +0.107 |

Knowledge inheritance is the single biggest jump. 
That's the point — the gain comes from stopping interference, 
not from a better optimizer.

## Citation

```bibtex
@inproceedings{jaffal2025pcdd,
  title={Federated Learning under Structural Class Absence 
         for Multi-City Transportation Mode Detection},
  author={Jaffal, Carla and El Hajj, Khalil and Sarieddeen, Hadi},
  booktitle={IEEE Globecom},
  year={2025}
}
```
## Authors

Carla Jaffal — American University of Beirut  
Khalil El Hajj — American University of Beirut  
Hadi Sarieddeen — American University of Beirut
