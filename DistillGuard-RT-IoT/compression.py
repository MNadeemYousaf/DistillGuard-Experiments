import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
import os
import pickle
import time
import copy
import evaluate
from models import StudentDense

# Configuration
DATA_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-RT-IoT/data"
CHECKPOINT_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-RT-IoT/checkpoints"
DEVICE = torch.device("cpu") # Quantization often best on CPU or specific backend
evaluate.DEVICE = DEVICE # Force evaluate to use CPU
torch.backends.quantized.engine = 'qnnpack'

def load_student():
    # Load meta
    with open(os.path.join(DATA_DIR, 'encoders.pkl'), 'rb') as f:
        meta = pickle.load(f)
        
    num_features = len(meta['net_cols']) + len(meta['temp_cols'])
    num_classes = len(meta['y_le'].classes_)
    
    model = StudentDense(num_features, num_classes, 256)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "student_best.pth"), map_location='cpu'))
    
    return model, meta

def load_test_data():
    datasets = torch.load(os.path.join(DATA_DIR, 'rt_iot_datasets.pt'), weights_only=False)
    test_loader = DataLoader(datasets['test'], batch_size=64, shuffle=False)
    return test_loader

def measure_size_and_inference(model, test_loader):
    # Size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    # Inference Time
    model.eval()
    start_time = time.time()
    total_samples = 0
    with torch.no_grad():
        for x_net, x_temp, y in test_loader:
            x_cat = torch.cat([x_net, x_temp], dim=1)
            # Ensure model and data are on CPU (evaluate.DEVICE set globally but explicitly checking here)
            _ = model(x_cat)
            total_samples += y.size(0)
    end_time = time.time()
    
    avg_inf_time = (end_time - start_time) / total_samples
    
    print(f"Model Size: {size_all_mb:.2f} MB")
    print(f"Avg Inference Time per Sample: {avg_inf_time*1000:.4f} ms")
    return size_all_mb, avg_inf_time

def apply_pruning(model, amount=0.3):
    print(f"\nApplying L1 Unstructured Pruning (Amount: {amount})...")
    # Prune Linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight') # Make permanent
    return model

def apply_quantization(model):
    print("\nApplying Dynamic Quantization (Int8)...")
    # Quantize Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

if __name__ == "__main__":
    test_loader = load_test_data()
    student, meta = load_student()
    class_names = meta['y_le'].classes_
    
    print("--- ORIGINAL STUDENT ---")
    orig_size, orig_time = measure_size_and_inference(student, test_loader)
    evaluate.evaluate_performance(student, test_loader, class_names)
    
    # Pruning
    pruned_student = copy.deepcopy(student)
    pruned_student = apply_pruning(pruned_student, amount=0.4) # 40% pruning
    print("--- PRUNED STUDENT ---")
    pruned_size, pruned_time = measure_size_and_inference(pruned_student, test_loader)
    evaluate.evaluate_performance(pruned_student, test_loader, class_names)
    
    # Quantization
    # Note: Quantization usually done on CPU
    quantized_student = apply_quantization(student) # Quantize original (or pruned)
    print("--- QUANTIZED STUDENT ---")
    quant_size, quant_time = measure_size_and_inference(quantized_student, test_loader)
    # Evaulate requires matching device (CPU) which is set globally
    evaluate.evaluate_performance(quantized_student, test_loader, class_names)
    
    print("\n--- COMPRESSION SUMMARY ---")
    print(f"Original: {orig_size:.2f} MB | {orig_time*1000:.3f} ms/sample")
    print(f"Pruned:   {pruned_size:.2f} MB | {pruned_time*1000:.3f} ms/sample (Sparse tensors needed for actual speedup)")
    print(f"Quantized:{quant_size:.2f} MB | {quant_time*1000:.3f} ms/sample")
    print(f"Reduction:{orig_size/quant_size:.1f}x Size")
