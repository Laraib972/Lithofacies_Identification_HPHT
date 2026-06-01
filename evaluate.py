import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from models import BiLSTM, BiGRU, ResASPPUnet       # Import your custom models

# ======================================
# 1. Overlapping Reconstruction Logic
# ======================================
def reconstruct_predictions(patch_probs, target_length, patch_size, stride):
    """
    Reconstructs continuous well log predictions from overlapping patches, 
    averaging the probabilities where patches overlap (Overhang Clipping Logic).
    """
    num_patches = patch_probs.shape[0]
    num_classes = patch_probs.shape[-1]
    
    # Initialize accumulators
    summed_probs = np.zeros((target_length, num_classes))
    counts = np.zeros((target_length, 1))

    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        
        patch_slice_start = 0
        patch_slice_end = patch_size
        
        # CLIP: If the patch goes past the end of the original well
        if end_idx > target_length:
            overhang = end_idx - target_length
            end_idx = target_length
            patch_slice_end = patch_size - overhang
            
            # Safety check
            if start_idx >= target_length:
                break

        # Accumulate Probabilities
        summed_probs[start_idx:end_idx] += patch_probs[i, patch_slice_start:patch_slice_end, :]
        counts[start_idx:end_idx] += 1

    # Avoid division by zero
    counts[counts == 0] = 1 
    
    # Average and take Argmax
    avg_probs = summed_probs / counts
    final_predictions = np.argmax(avg_probs, axis=-1)
    
    return final_predictions

# ======================================
# 2. Generalized Evaluation Pipeline
# ======================================
def evaluate_blind_well(model, x_blind_patched, y_blind_encoded, original_length, 
                        patch_size, stride, class_names, device, model_name, plot_cm=True):
    """
    Runs inference on a blind well, reconstructs the continuous sequence, 
    and generates robust evaluation metrics excluding zero-count classes.
    """
    print(f"\n--- Evaluating {model_name} on Blind Well (Length: {original_length}) ---")
    
    # Ensure input is a tensor and on the correct device
    if not torch.is_tensor(x_blind_patched):
        x_blind_tensor = torch.tensor(x_blind_patched, dtype=torch.float32).to(device)
    else:
        x_blind_tensor = x_blind_patched.to(device)

    # Get Softmax Probabilities
    model.eval()
    with torch.no_grad():
        logits = model(x_blind_tensor)  
        probs = F.softmax(logits, dim=-1).cpu().numpy()

    # Run Reconstruction using Overlap/Clipping
    predicted_labels = reconstruct_predictions(probs, original_length, patch_size, stride)
    
    # Sanity Check
    assert predicted_labels.shape[0] == original_length, f"Shape mismatch! Expected {original_length}, got {predicted_labels.shape[0]}"
    
    # Prepare True Labels
    if torch.is_tensor(y_blind_encoded):
        y_blind_encoded = y_blind_encoded.cpu().numpy()
        
    if y_blind_encoded.ndim > 1 and y_blind_encoded.shape[-1] > 1:
        true_labels = np.argmax(y_blind_encoded, axis=-1)
    else:
        true_labels = y_blind_encoded.flatten()

    # Robust Metric Calculation (Handling Zero-Count Classes)
    present_classes = np.unique(true_labels)
    present_class_names = [class_names[i] for i in present_classes]

    accuracy = accuracy_score(true_labels, predicted_labels)
    
    # Explicitly calculate metrics only on classes present in this specific well
    report = classification_report(
        true_labels, 
        predicted_labels, 
        labels=present_classes,
        target_names=present_class_names, 
        digits=4, 
        zero_division=0
    )
    
    macro_f1 = f1_score(
        true_labels, 
        predicted_labels, 
        labels=present_classes, 
        average='macro', 
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1-score (macro): {macro_f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    # Plot Confusion Matrix
    if plot_cm:
        cm = confusion_matrix(true_labels, predicted_labels, labels=present_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=present_class_names, 
                    yticklabels=present_class_names)
        plt.title(f'Confusion Matrix: {model_name} (Present Classes Only)')
        plt.ylabel('True Lithofacies')
        plt.xlabel('Predicted Lithofacies')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save and show
        plot_filename = f"{model_name}_blind_cm.png"
        plt.savefig(plot_filename)
        print(f"Saved confusion matrix plot to {plot_filename}")
        plt.show()

    return predicted_labels, accuracy, macro_f1

# ======================================
# 3. Command Line Argument Parser
# ======================================
def parse_args():
    parser = argparse.ArgumentParser(description="Blind Well Evaluation Script (Overlapping Windows)")
    
    parser.add_argument("--model", type=str, required=True, 
                        choices=["Bi-GRU", "Bi-LSTM", "Res-ASPP-UNet"],
                        help="Select the base architecture to evaluate.")
    
    parser.add_argument("--use_ltn", action="store_true", 
                        help="Flag if evaluating the LTN-enhanced model.")
    
    parser.add_argument("--weights", type=str, required=True, 
                        help="Path to the saved .pth weights file for this model.")
    
    parser.add_argument("--stride", type=int, default=100,
                        help="Stride used during data patching.")
    
    return parser.parse_args()

# ======================================
# 4. Main Execution Block
# ======================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # --- Hyperparameters based on Model Table ---
    hidden_size = 64
    num_layers = 1
    input_size = 8
    output_size = 9

    # Dynamic dropout & patch size mapping
    if args.model == "Bi-GRU":
        dropout = 0.0
        patch_size = 150
    elif args.model == "Bi-LSTM":
        dropout = 0.1
        patch_size = 150
    elif args.model == "Res-ASPP-UNet":
        dropout = 0.1 if args.use_ltn else 0.0
        patch_size = 160
        
    stride = args.stride

    model_display_name = f"{args.model}-LTN" if args.use_ltn else args.model

    class_names = [
        "Marine_Claystone/Shale", "Carbonate", "Carb_Shale",
        "Marine_Sandstone/Shaly_Sand", "Marine_Claystone/Shale_KT",
        "Transitional_Claystone/Shale", "Fluvial_Claystone/Shale",
        "Fluvial_Shaly_Sand", "Fluvial_Sandstone"
    ]

    # --- Load Blind Data ---
    # NOTE: Replace these placeholders with your actual blind well data loading logic
    # x_blind_patched = ... 
    # y_blind_encoded = ...
    # original_length = 32845
    
    # --- Instantiate Model ---
    if args.model == "Bi-GRU":
        model = BiGRU(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    elif args.model == "Bi-LSTM":
        model = BiLSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    elif args.model == "Res-ASPP-UNet":
        model = ResASPPUnet(input_features=input_size, output_classes=output_size, dropout_rate=dropout).to(device)

    # Load Weights
    print(f"Loading weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # --- Execute Evaluation ---
    # evaluate_blind_well(
    #     model=model, 
    #     x_blind_patched=x_blind_patched, 
    #     y_blind_encoded=y_blind_encoded, 
    #     original_length=original_length, 
    #     patch_size=patch_size, 
    #     stride=stride, 
    #     class_names=class_names, 
    #     device=device,
    #     model_name=model_display_name,
    #     plot_cm=True
    # )

if __name__ == "__main__":
    main()