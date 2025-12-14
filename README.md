# DistillGuard: Comparative Study on IoT Intrusion Detection

This repository contains the implementation and results of the **DistillGuard** framework (Knowledge Distillation for IoT Security) applied to three major datasets:

1.  **ToN-IoT** (Heterogeneous)
2.  **RT-IoT** (Real-Time)
3.  **N-BaIoT** (Botnet Traffic)

## Project Structure

*   `DistillGuard-ToN-IoT/`: Implementation for ToN-IoT.
*   `DistillGuard-RT-IoT/`: Implementation for RT-IoT.
*   `DistillGuard-N-BaIoT/`: Implementation for N-BaIoT.

## Key Results

| Dataset | Teacher Acc | Student Acc | Compression | Robustness |
| :--- | :--- | :--- | :--- | :--- |
| **ToN-IoT** | 96% | 95.9% | 4x | Low |
| **RT-IoT** | 99% | 99.5% | 4x | Low |
| **N-BaIoT** | 17% (Fail) | 82.6% (Success) | 4x | Low |

See `DistillGuard-N-BaIoT/comparison_study.pdf` for the full report.

## Usage

Each directory contains its own `README.md` and scripts:
1.  `data_processor.py`
2.  `train_distillguard.py`
3.  `evaluate.py`

## Requirements
*   PyTorch
*   Pandas
*   Scikit-Learn
*   Matplotlib
