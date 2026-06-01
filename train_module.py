import torch

# Assuming your loss functions are in a utils file
from utils import total_loss_function, dual_focal_loss

def accuracy_from_logits(logits, y_true):
    """Calculates categorical accuracy from raw logits."""
    preds = torch.argmax(logits, dim=-1)
    labels = torch.argmax(y_true, dim=-1)
    correct = (preds == labels).float()
    return correct.mean().item()

def evaluate(model, val_loader, device):
    """Evaluates the model on the validation set using only focal loss (no LTN)."""
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)

            loss = dual_focal_loss(logits, y_batch)
            acc = accuracy_from_logits(logits, y_batch)

            total_loss += loss.item()
            total_acc += acc

    return total_loss / len(val_loader), total_acc / len(val_loader)

def train(model, train_loader, val_loader, optimizer, rules, class_names, device,
          num_epochs=70, lambda_ltn=0.0):
    """
    Main training loop combining Dual Focal Loss and Canonical LTN Loss.
    """
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        
        for x_batch, y_batch, env_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            env_batch = env_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(x_batch)

            # Training uses focal loss + LTN penalty
            loss = total_loss_function(
                logits, y_batch, env_batch, class_names, rules, lambda_ltn
            )
            acc = accuracy_from_logits(logits, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc

        train_loss = total_loss / len(train_loader)
        train_acc  = total_acc / len(train_loader)

        val_loss, val_acc = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history