import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle

from models import StudentDense, TeacherTransformer

# Configuration
DATA_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-ToN-IoT/data"
CHECKPOINT_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-ToN-IoT/checkpoints"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_test_data():
    datasets = torch.load(os.path.join(DATA_DIR, 'ton_iot_datasets.pt'), weights_only=False)
    test_loader = DataLoader(datasets['test'], batch_size=64, shuffle=False)
    
    with open(os.path.join(DATA_DIR, 'encoders.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return test_loader, meta

def evaluate_performance(model, test_loader, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_net, x_temp, y in test_loader:
            x_net, x_temp = x_net.to(DEVICE), x_temp.to(DEVICE)
            # Student inputs concatenated? Or handle both?
            # StudentDense takes ONE input (cat). Teacher takes TWO.
            # We need to distinguish.
            if isinstance(model, StudentDense):
                x = torch.cat([x_net, x_temp], dim=1)
                logits, _ = model(x)
            else:
                logits, _ = model(x_net, x_temp)
                
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    # Metrics
    print("Classification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    return report

def fgsm_attack(model, data, epsilon, target=None):
    # Generates Adversarial Example
    # data: inputs
    # target: true labels
    # Model must have gradients enabled wrt input
    
    # For Student: input is x_cat
    # For Teacher: input is (x_net, x_temp) - Harder to attack split inputs easily via standard FGSM
    # We focus on Student robustness (DistillGuard goal: Student learns robustness)
    
    input_tensor = data.clone().detach().to(DEVICE)
    input_tensor.requires_grad = True
    
    logits, _ = model(input_tensor)
    if target is None:
        target = logits.argmax(dim=1) # Untargeted
        
    loss = nn.CrossEntropyLoss()(logits, target)
    model.zero_grad()
    loss.backward()
    
    data_grad = input_tensor.grad.data
    sign_data_grad = data_grad.sign()
    
    perturbed_image = input_tensor + epsilon * sign_data_grad
    # Clamp if normalized [0,1]?
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def test_robustness(model, test_loader, epsilons=[0, 0.05, 0.1, 0.2, 0.3]):
    print("\n--- Adversarial Robustness (FGSM) ---")
    accuracies = []
    
    for eps in epsilons:
        correct = 0
        total = 0
        
        for x_net, x_temp, y in test_loader:
            x_net, x_temp, y = x_net.to(DEVICE), x_temp.to(DEVICE), y.to(DEVICE)
            
            # Combine for student
            x_cat = torch.cat([x_net, x_temp], dim=1)
            
            if eps == 0:
                perturbed = x_cat
            else:
                # Generate Attack
                perturbed = fgsm_attack(model, x_cat, eps, y)
                
            with torch.no_grad():
                logits, _ = model(perturbed)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        acc = correct / total
        accuracies.append(acc)
        print(f"Epsilon: {eps}\tTest Accuracy = {acc:.4f}")
        
    plt.figure()
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("FGSM Robustness (Student)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig('robustness_curve.png')

def explain_gradient(model, sample_input, class_names, feature_names):
    # Gradient Heatmap for one sample
    input_tensor = sample_input.clone().detach().to(DEVICE).unsqueeze(0)
    input_tensor.requires_grad = True
    
    logits, _ = model(input_tensor)
    score = logits[0, logits.argmax()]
    score.backward()
    
    gradients = input_tensor.grad.data.abs().cpu().numpy()[0]
    
    # Plot top 10 features
    indices = np.argsort(gradients)[-10:]
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), gradients[indices])
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.title('Top 10 Feature Gradients')
    plt.xlabel('Gradient Magnitude')
    plt.tight_layout()
    plt.savefig('gradient_explanation.png')


if __name__ == "__main__":
    test_loader, meta = load_test_data()
    class_names = meta['y_le'].classes_
    feature_names = meta['net_cols'] + meta['temp_cols']
    num_features = len(feature_names)
    num_classes = len(class_names)
    
    # Load Student (Distilled)
    print("Loading Student Model...")
    student = StudentDense(num_features, num_classes, 256).to(DEVICE)
    try:
        student.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "student_best.pth"), map_location=DEVICE))
    except FileNotFoundError:
        print("Student checkpoint not found. Ensure training finished.")
        exit()
        
    print("\n--- Performance Evaluation ---")
    evaluate_performance(student, test_loader, class_names)
    
    # Robustness
    test_robustness(student, test_loader)
    
    # Explainability
    # Pick a sample
    x_n, x_t, _ = test_loader.dataset[0]
    x_cat = torch.cat([x_n, x_t], dim=0) # Flat for one sample
    explain_gradient(student, x_cat, class_names, feature_names)
    
    print("\nEvaluation Complete. Check artifacts.")
