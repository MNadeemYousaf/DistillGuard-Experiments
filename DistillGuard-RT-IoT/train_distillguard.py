import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

from models import TeacherTransformer, StudentDense
from distiller import SGKDistiller

# Configuration
DATA_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-RT-IoT/data"
CHECKPOINT_DIR = "/Users/nadeemyousaf/.gemini/antigravity/scratch/DistillGuard-RT-IoT/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using Device: {DEVICE}")

def load_data():
    dataset_path = os.path.join(DATA_DIR, 'rt_iot_datasets.pt')
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Run data_processor.py first.")
    
    datasets = torch.load(dataset_path, weights_only=False)
    # Batch sizes from paper: Teacher 32, Student 64
    train_loader_t = DataLoader(datasets['train'], batch_size=32, shuffle=True)
    val_loader_t = DataLoader(datasets['val'], batch_size=32, shuffle=False)
    
    train_loader_s = DataLoader(datasets['train'], batch_size=64, shuffle=True)
    val_loader_s = DataLoader(datasets['val'], batch_size=64, shuffle=False)
    
    return train_loader_t, val_loader_t, train_loader_s, val_loader_s, datasets

def train_teacher(model, train_loader, val_loader, epochs=50, lr=0.001):
    print("\n" + "="*40)
    print("TRAINING TEACHER TRANSFORMER")
    print("="*40)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x_net, x_temp, y in loop:
            x_net, x_temp, y = x_net.to(DEVICE), x_temp.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(x_net, x_temp) # Ignore hidden feat here
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        train_acc = correct / total
        train_loss = total_loss / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_net, x_temp, y in val_loader:
                x_net, x_temp, y = x_net.to(DEVICE), x_temp.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x_net, x_temp)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "teacher_best.pth"))
            print(f"--> Saved Best Teacher ({best_acc:.4f})")
            
    print("Teacher Training Complete.")
    return model

def train_student_sgkd(student, teacher, train_loader, val_loader, epochs=50, lr=0.0005):
    print("\n" + "="*40)
    print("TRAINING STUDENT (SG-KD)")
    print("="*40)
    
    # Load best teacher
    teacher.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "teacher_best.pth")))
    teacher.to(DEVICE)
    teacher.eval() # Freeze teacher
    
    distiller = SGKDistiller(teacher, student, alpha=0.5, beta=0.5, retention_ratio=0.3)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_acc = 0.0
    history = {'ce_loss': [], 'distill_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        student.train()
        total_ce = 0
        total_sgkd = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Student]", leave=False)
        for x_net, x_temp, y in loop:
            x_net, x_temp, y = x_net.to(DEVICE), x_temp.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Compute SG-KD Loss (Forward/Backward logic handled inside)
            loss, ce, sgkd = distiller.compute_loss(x_net, x_temp, y)
            
            loss.backward()
            optimizer.step()
            
            # Monitoring based on Student Logits
            # Need to re-forward student purely for acc? 
            # distiller already did it, but capturing it inside compute_loss is messy unless we return logits.
            # Let's do a quick inference or just rely on CE part which implies accuracy.
            # To be precise, let's just forward student again (cheap) or modify distiller to return logits.
            # Modification is better but let's just do quick forward for stats.
            with torch.no_grad():
                x_cat = torch.cat([x_net, x_temp], dim=1)
                s_logits, _ = student(x_cat)
                preds = s_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            total_ce += ce
            total_sgkd += sgkd
            
            loop.set_postfix(ce=ce, sgkd=sgkd)
            
        train_acc = correct / total
        avg_ce = total_ce / len(train_loader)
        avg_sgkd = total_sgkd / len(train_loader)
        
        # Validation
        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_net, x_temp, y in val_loader:
                x_net, x_temp, y = x_net.to(DEVICE), x_temp.to(DEVICE), y.to(DEVICE)
                x_cat = torch.cat([x_net, x_temp], dim=1)
                logits, _ = student(x_cat)
                preds = logits.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1} | CE Loss: {avg_ce:.4f} | SGKD Loss: {avg_sgkd:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        history['ce_loss'].append(avg_ce)
        history['distill_loss'].append(avg_sgkd)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), os.path.join(CHECKPOINT_DIR, "student_best.pth"))
            
    # Plot history
    plt.figure()
    plt.plot(history['ce_loss'], label='CE Loss')
    plt.plot(history['distill_loss'], label='Distill Loss')
    plt.legend()
    plt.savefig('student_training_losses.png')
    
    return student

# --- FGSM Attack (Copied for Training) ---
def fgsm_attack_train(model, x_net, x_temp, y, epsilon):
    # Enable grad for inputs
    x_net = x_net.clone().detach().requires_grad_(True)
    x_temp = x_temp.clone().detach().requires_grad_(True)
    
    logits, _ = model(x_net, x_temp)
    loss = nn.CrossEntropyLoss()(logits, y)
    
    model.zero_grad()
    loss.backward()
    
    # Perturb both inputs
    # Note: Normalized features [0,1]. Clamp after.
    x_net_adv = x_net + epsilon * x_net.grad.sign()
    x_temp_adv = x_temp + epsilon * x_temp.grad.sign()
    
    x_net_adv = torch.clamp(x_net_adv, 0, 1)
    x_temp_adv = torch.clamp(x_temp_adv, 0, 1)
    
    return x_net_adv.detach(), x_temp_adv.detach()

def fine_tune_teacher_adversarial(model, train_loader, epochs=10, epsilon=0.1, lr=0.0001):
    print("\n" + "="*40)
    print("STEP 7: ADVERSARIAL TRAINING (TEACHER)")
    print("="*40)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Adv Epoch {epoch+1}/{epochs}", leave=False)
        for x_net, x_temp, y in loop:
            x_net, x_temp, y = x_net.to(DEVICE), x_temp.to(DEVICE), y.to(DEVICE)
            
            # Generate Adversarial Samples
            # Model needs to be in a state to compute grads wrt input, but we want to update weights too.
            # Standard way: Compute Adv samples first, then train on them.
            x_net_adv, x_temp_adv = fgsm_attack_train(model, x_net, x_temp, y, epsilon)
            
            # Training on Mixed Batch? Or just Adv?
            # Paper: "Fine-tune using both clean and adversarial samples".
            # Let's concatenate them.
            x_net_combined = torch.cat([x_net, x_net_adv], dim=0)
            x_temp_combined = torch.cat([x_temp, x_temp_adv], dim=0)
            y_combined = torch.cat([y, y], dim=0)
            
            # Shuffle?
            perm = torch.randperm(y_combined.size(0))
            x_net_combined = x_net_combined[perm]
            x_temp_combined = x_temp_combined[perm]
            y_combined = y_combined[perm]
            
            optimizer.zero_grad()
            logits, _ = model(x_net_combined, x_temp_combined)
            loss = criterion(logits, y_combined)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y_combined.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_combined).sum().item()
            total += y_combined.size(0)
            
            loop.set_postfix(loss=loss.item())
            
        print(f"Adv Epoch {epoch+1} | Loss: {total_loss/total:.4f} | Acc: {correct/total:.4f}")
        
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "teacher_adversarial.pth"))
    print("Teacher Adversarial Fine-tuning Complete.")
    return model

if __name__ == "__main__":
    t_loader, t_val, s_loader, s_val, datasets = load_data()
    
    # Get Dims
    sample_net, sample_temp, sample_y = datasets['train'][0]
    NUM_CLASSES = len(torch.unique(datasets['train'][:][2]))
    
    with open(os.path.join(DATA_DIR, 'encoders.pkl'), 'rb') as f:
        meta = pickle.load(f)
        NUM_CLASSES = len(meta['y_le'].classes_)
    
    print(f"Classes: {NUM_CLASSES}")
    
    # 1. Initialize Teacher
    teacher = TeacherTransformer(
        net_input_dim=sample_net.shape[0],
        temp_input_dim=sample_temp.shape[0],
        num_classes=NUM_CLASSES,
        hidden_dim=512,
        layers=6, 
        heads=8
    ).to(DEVICE)
    
    # 2. Train Teacher (Standard)
    # Check if pretrained exists to skip
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "teacher_best.pth")):
        print("Loading pre-trained teacher...")
        teacher.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "teacher_best.pth")))
    else:
        teacher = train_teacher(teacher, t_loader, t_val, epochs=50)

    # 3. Fine-tune Teacher (Adversarial) - Step 7
    # Use epsilon=0.1 as per paper range 0.01-0.1
    teacher = fine_tune_teacher_adversarial(teacher, t_loader, epochs=5, epsilon=0.05)
    
    # 4. Initialize Student
    student = StudentDense(
        total_input_dim=sample_net.shape[0] + sample_temp.shape[0],
        num_classes=NUM_CLASSES,
        hidden_units=256
    ).to(DEVICE)
    
    # 5. Train Student (Distillation) using Adversarial Teacher?
    # Usually we use the robust teacher to distill.
    student = train_student_sgkd(student, teacher, s_loader, s_val, epochs=50)
