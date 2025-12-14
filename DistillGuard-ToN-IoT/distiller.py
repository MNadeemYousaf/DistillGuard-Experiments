import torch
import torch.nn as nn
import torch.nn.functional as F

class SGKDistiller:
    def __init__(self, teacher, student, alpha=0.5, beta=0.5, retention_ratio=0.3):
        """
        Selective Gradient Knowledge Distillation (SG-KD) Manager.
        
        Args:
            teacher: Pre-trained Teacher Model
            student: Student Model
            alpha: Weight for Student CrossEntropy Loss
            beta: Weight for Distillation Loss
            retention_ratio: Top-K ratio (e.g., 0.3 means keep top 30% most important neurons)
        """
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.retention_ratio = retention_ratio
        
    def get_gradient_mask(self, param_grad):
        """
        Computes the mask M based on Top-K gradients.
        param_grad: Gradient of loss w.r.t the hidden representation [Batch, Hidden]
        """
        # Calculate importance score: absolute gradient
        importance = torch.abs(param_grad)
        
        # Determine K
        batch_size, hidden_dim = importance.shape
        k = int(hidden_dim * self.retention_ratio)
        
        # Find Top-K indices per sample or global? 
        # "Select Top-K gradient-important neurons". Usually per-sample or average.
        # Let's assume per-sample dynamism (DistillGuard likely implies dynamic selection).
        
        topk_values, _ = torch.topk(importance, k=k, dim=1)
        # Threshold is the k-th value
        threshold = topk_values[:, -1].unsqueeze(1)
        
        # Mask: 1 if importance >= threshold, else 0
        mask = (importance >= threshold).float()
        return mask

    def compute_loss(self, x_net, x_temp, y):
        """
        Performs the forward/backward pass to compute SG-KD Loss.
        """
        # 1. Teacher Forward Pass
        # We need gradients w.r.t Teacher's Hidden Representation
        self.teacher.eval() # Teacher is fixed
        
        # Enable grad for input/intermediates just to get gradients
        # But Teacher weights are frozen.
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        # Hook or Manual Forward to capture features
        # models.py returns (logits, features)
        
        # We need gradients on the 'features'.
        # Since Teacher is in eval/no_grad usually, we must enable grad for just the features.
        
        # Forward inputs
        net_emb = self.teacher.net_proj(x_net).unsqueeze(1)
        temp_emb = self.teacher.temp_proj(x_temp).unsqueeze(1)
        fused = self.teacher.hybrid_attn(net_emb, temp_emb)
        out_enc = self.teacher.transformer_encoder(fused)
        teacher_feat = out_enc.squeeze(1).detach() # Detached first? No, we need grad ON this.
        
        teacher_feat.requires_grad = True # Enable grad for mask computation
        
        t_logits = self.teacher.classifier(teacher_feat)
        
        # Teacher Loss (CE)
        t_loss = F.cross_entropy(t_logits, y)
        
        # 2. Compute Gradients w.r.t Teacher Hidden features
        # Note: We clear existing grads first
        self.teacher.zero_grad()
        t_loss.backward()
        
        grad_hidden = teacher_feat.grad # [Batch, Hidden]
        
        # 3. Compute Mask
        mask = self.get_gradient_mask(grad_hidden)
        
        # 4. Student Forward Pass
        # Student inputs: Concatenate network + temporal
        x_student = torch.cat([x_net, x_temp], dim=1)
        s_logits, s_feat_projected = self.student(x_student)
        
        # 5. Losses
        # Student CE Loss
        loss_ce = F.cross_entropy(s_logits, y)
        
        # Distillation Loss (MSE on Masked Features)
        # L_SGKD = || M * (H_t - H_s) ||^2
        # Note: teacher_feat should be treated as constant target now, detach it.
        # But in step 1 we needed grad. For usage in Loss, we detach.
        target_feat = teacher_feat.detach()
        
        # Apply mask
        diff = (target_feat - s_feat_projected) * mask
        loss_sgkd = (diff ** 2).sum() / mask.sum() # Mean over active neurons
        
        total_loss = self.alpha * loss_ce + self.beta * loss_sgkd
        
        return total_loss, loss_ce.item(), loss_sgkd.item()
