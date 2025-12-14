# DistillGuard-RT-IoT Experiment

This project applies the **DistillGuard** framework (Algorithm 1) to the **RT-IoT2022** dataset.

## Dataset Details
*   **Source**: RT-IoT2022 (Csv).
*   **Samples**: ~123,000.
*   **Features**: 
    *   **Network**: `id.orig_p`, `id.resp_p`, `proto`, `service` (Note: IPs were absent in provided file).
    *   **Temporal**: Top 15 features selected via Mutual Information (e.g., `flow_iat`, `window_size`).
*   **Classes**: 12 (Attack types + Normal).

## Implementation
Identical to the ToN-IoT implementation, strictly following Algorithm 1:
1.  **Teacher**: Transformer with Hybrid Attention.
2.  **Adversarial Training**: Teacher fine-tuned on FGSM samples.
3.  **Student**: Distilled via SG-KD.
4.  **Compression**: Pruning + Quantization.

## Usage
1.  **Process Data**: `python3 data_processor.py`
2.  **Train**: `python3 train_distillguard.py`
3.  **Compress**: `python3 compression.py`
4.  **Evaluate**: `python3 evaluate.py`

## Results
Results will be generated in this directory as PNG files (`confusion_matrix.png`, etc.) upon completion.
