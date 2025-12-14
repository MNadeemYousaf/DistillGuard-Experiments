# DistillGuard-ToN-IoT Implementation

This project implements the **DistillGuard** framework (Transformer Teacher + Dense Student + SG-KD) on the **ToN-IoT** dataset, following the specifications of the referenced paper.

## Project Structure
*   `data_processor.py`: Data pipeline (Cleaning, MRMR-based Feature Selection, Normalization, Splitting).
*   `models.py`: Model Definitions.
    *   **Teacher**: Transformer with Hybrid Attention (Network + Temporal features).
    *   **Student**: Lightweight Dense Network.
*   `distiller.py`: **Selective Gradient Knowledge Distillation (SG-KD)** logic.
*   `train_distillguard.py`: Main training script (Teacher -> Student).
*   `evaluate.py`: Evaluation suite (Metrics, FGSM Robustness, Gradient Heatmaps).
*   `checkpoints/`: Directory for saved models.
*   `data/`: Directory for processed datasets (`ton_iot_datasets.pt`).

## Implementation Details (Aligned with Paper)
### 1. Data Pipeline
*   **Dataset**: ToN-IoT Network Flows (211k samples).
*   **Cleaning**: Dropped irrelevant columns, handled NaNs.
*   **Selection**: Top 21 features (6 Network + 15 Temporal) using Mutual Information.
*   **Encoding**: Label Encoding for Network IPs/Ports, MinMax Scaling for all.

### 2. Models
*   **Teacher**: 6-Layer Transformer, 8 Heads, 512 Hidden Unit. Implementation of **Hybrid Attention** via Cross-Attention between Network and Temporal embeddings.
*   **Student**: 3-Layer Dense Network (256 Hidden).
*   **Distillation**: Uses **SG-KD** (Top-30% Gradient Mask) to distill knowledge from Teacher to Student.

### 3. Hyperparameters
*   **Epochs**: 50 (Teacher) + 50 (Student).
*   **Teacher LR**: 0.001.
*   **Student LR**: 0.0005.
*   **Batch Size**: 32 (Teacher), 64 (Student).
*   **FGSM Epsilon**: 0.05 (Training & Eval).

## Usage
1.  **Process Data**:
    ```bash
    python3 data_processor.py
    ```
2.  **Train Models** (Steps 6-8):
    ```bash
    python3 train_distillguard.py
    ```
    *Includes Step 7: Adversarial Fine-tuning of Teacher.*
3.  **Compress Models** (Step 9):
    ```bash
    python3 compression.py
    ```
    *Applies Pruning and Quantization.*
4.  **Evaluate** (Step 10):
    ```bash
    python3 evaluate.py
    ```
    Outputs:
    *   `confusion_matrix.png`
    *   `robustness_curve.png` (FGSM)
    *   `gradient_explanation.png` (XAI)
    *   Console Classification Report.

## Results (Expected)
The framework aims to achieve high accuracy (Teacher-level) with a lightweight Student model, while demonstrating robustness against adversarial attacks (FGSM).
