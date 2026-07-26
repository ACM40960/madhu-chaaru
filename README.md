# Adversarial Robustness Testing & Trust Safety Pipeline for Identity Document Verification

MSc Data and Computational Science — Mathematical Modelling Module
- Student 1: Madhumitha Vaithiyanathan
- Student 2: Chaarumathy Mathiyalaghan
- Supervisor: Dr. Sarp Akcay

## Project Overview

This project builds a three-class identity document verification system that:
- Distinguishes between **real** documents, **benign** user errors, and **malicious** fraud tampering
- Systematically maps the exact breaking points where AI verification fails
- Trains a hardened classifier robust to both physical and adversarial perturbations
- Deploys a Trust & Safety pipeline with explainable GradCAM visualisations

## Repository Structure

- `notebooks/` — Jupyter notebooks for each project phase
- `outputs/` — Generated plots and evaluation results
- `data/` — Dataset splits and frame index CSVs
- `requirements.txt` — All required Python libraries

## Dataset

- **MIDV-500** — 50 identity document types (Burie et al., 2015)
- **MIDV-2019** — 50 identity document types, challenging conditions (Burie et al., 2019)
- Both datasets are publicly available and contain only specimen documents

## Three-Class System

| Class | Label | Definition |
|---|---|---|
| Real | 0 | Genuine unaltered document |
| Benign | 1 | User error — blur, rotation, bad lighting |
| Malicious | 2 | Deliberate fraud — face swap, text alteration |

## Tech Stack

Python, PyTorch, EfficientNet-B0, ResNet18, Albumentations, Foolbox, IBM ART, GradCAM, OpenCV, Scikit-learn
