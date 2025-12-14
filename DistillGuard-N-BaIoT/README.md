# DistillGuard-N-BaIoT Experiment

This project applies the **DistillGuard** framework (Algorithm 1) to the full **N-BaIoT** dataset (All 9 Devices).

## Dataset Details
*   **Source**: N-BaIoT (UCI / Kaggle).
*   **Original Size**: ~7 Million rows (across ~89 CSVs).
*   **Used Size**: 211,873 samples (Subsampled Stratified) for training efficiency on M1.
*   **Classes**: Multiple Botnet families (Mirai, Gafgyt) across 9 IoT devices + Benign.
*   **Features**: 115 Flow Statistics.
    *   **Network (Short-term context)**: Features with `L1` or `L3` decay (approx immediate traffic).
    *   **Temporal (Long-term context)**: Features with `L5`, `L0.1`, `L0.01` decay.

## Implementation
Strict adherence to Algorithm 1:
1.  **Teacher**: Transformer with Hybrid Attention (Net vs Temp).
2.  **Adversarial Training**: Teacher fine-tuned on FGSM ($\epsilon=0.05$).
3.  **Student**: Distilled via SG-KD.
4.  **Compression**: Pruning + Quantization.

## Usage
1.  **Process Data**: `python3 data_processor.py` (Handles merging & subsampling).
2.  **Train**: `python3 train_distillguard.py`
3.  **Compress**: `python3 compression.py`
4.  **Evaluate**: `python3 evaluate.py`

## Results
Artifacts (`n_baiot_*.png`) and reports will be generated after execution.
