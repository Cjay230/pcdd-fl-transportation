# Federated Learning under Structural Class Absence for Multi-City Transportation Mode Detection
**Accepted at IEEE Globecom 2025**

## Overview
This repository contains the implementation for our paper on 
Partially Class-Disjoint Data (PCDD) in federated learning. 
We identify gradient interference from physically uninformed 
clients as the key failure mechanism when cities have 
structurally different transport mode sets, and propose a 
two-level class-aware FL framework to address it.

## Key Result
FedAvg collapses jog classification to F1 = 0.000 due to 
gradient interference from three cities that have never 
produced a jogging trip. Our method recovers it to F1 = 0.567.

## Method
- **Knowledge Inheritance**: freezes output nodes for absent classes locally
- **cwFedAvg**: aggregates each class only from cities that observed it
- **Class-Weighted Loss**: corrects within-city geographic imbalance
- **LAWA**: weights clients by training balance with γ=0.2

## Results
| Method | Macro F1 |
|--------|----------|
| FedAvg | 0.524 |
| FedProx | 0.522 |
| SCAFFOLD | 0.366 |
| FedBN | 0.624 |
| MOON | 0.560 |
| **Proposed E5** | **0.631** |

## Requirements
