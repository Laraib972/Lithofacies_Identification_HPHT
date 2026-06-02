import argparse
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import random
import os

# Assuming your models are accessible from the main directory
# from models import BiGRU, BiLSTM, ResASPPUnet

# ==========================================
# 1. REPRODUCIBILITY SETUP
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ==========================================
# 2. HELPER: GENERIC STITCHING FUNCTION
# ==========================================
def stitch_patches(patched_data, target_length, patch_size, stride):
    """
    Stitches patches back to original length, handling overlaps by averaging.
    Input: (Num_Patches, Patch_Size, Channels)
    Output: (Target_Length, Channels)
    """
    if torch.is_tensor(patched_data):
        patched_data = patched_data.cpu().numpy()
        
    num_patches = patched_data.shape[0]
    channels = patched_data.shape[-1]
    
    summed_data = np.zeros((target_length, channels))
    counts = np.zeros((target_length, 1))

    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        
        # Handle Edge Cases (End of Well)
        slice_len = patch_size
        if end_idx > target_length:
            diff = end_idx - target_length
            slice_len = patch_size - diff
            end_idx = target_length
            
        if start_idx >= target_length:
            break
            
        # Accumulate
        summed_data[start_idx:end_idx] += patched_data[i, :slice_len, :]
        counts[start_idx:end_idx] += 1

    # Average
    counts[counts == 0] = 1 # Avoid div by zero
    avg_data = summed_data / counts
    return avg_data

# ==========================================
# 3. HELPER: ADD NOISE
# ==========================================
def add_noise(tensor, level):
    """Adds Gaussian noise to the input tensor."""
    if level == 0.0:
        return tensor
    noise = torch.randn_like(tensor) * level
    return tensor + noise

# ==========================================
# 4. RUN SENSITIVITY ANALYSIS
# ==========================================
def run_global_noise_test(model, model_name, x_global_tensor, y_global_actual, 
                          patch_lengths, orig_lengths, noise_levels, patch_size, stride):
    """
    Evaluates noise robustness. Dynamically splits and stitches any number of blind wells.
    """
    results = []
    print(f"\nStarting Noise Robustness Test on Global Dataset ({len(y_global_actual)} samples) for {model_name}...")
    
    model.eval() # Set to evaluation mode
    
    # Iterate over noise levels
    for level in noise_levels:
        # 1. Add Noise
        x_noisy = add_noise(x_global_tensor, level)
        
        # 2. Predict
        with torch.no_grad():
            logits = model(x_noisy)
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            
            # --- DYNAMIC STITCHING LOGIC ---
            stitched_predictions = []
            start_idx = 0
            
            # Loop through the lengths to dynamically split and stitch
            for w_patch_len, w_orig_len in zip(patch_lengths, orig_lengths):
                # Extract well-specific chunk
                probs_w = probs[start_idx : start_idx + w_patch_len]
                
                # Stitch the well
                stitched_w = stitch_patches(probs_w, w_orig_len, patch_size, stride)
                
                # Take argmax and append
                stitched_predictions.append(np.argmax(stitched_w, axis=-1))
                
                # Move the index forward
                start_idx += w_patch_len
            
            # Concatenate all stitched predictions
            y_pred_global = np.concatenate(stitched_predictions)

        # 3. Calculate Metrics (Using your exact Macro F1 formulation)
        acc = accuracy_score(y_global_actual, y_pred_global)
        f1 = f1_score(y_global_actual, y_pred_global, average='macro')
        
        # 4. Store
        results.append({'Model': model_name, 'Noise Level': level, 'Metric': 'Global Accuracy', 'Score': acc})
        results.append({'Model': model_name, 'Noise Level': level, 'Metric': 'Global Macro F1', 'Score': f1})
        
        print(f"  {model_name} | Noise {level:.2f} -> Acc: {acc:.4f}, Macro F1: {f1:.4f}")

    df_noise = pd.DataFrame(results)
    print(f"✅ Experiment Complete for {model_name}.")
    return df_noise

# ==========================================
# 5. COMMAND LINE INTERFACE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Global Noise Sensitivity Analysis")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["Bi-GRU-LTN", "Bi-LSTM-LTN", "Res-ASPP-UNet-LTN", "Bi-GRU", "Bi-LSTM", "Res-ASPP-UNet"],
                        help="Select the architecture for Noise Analysis.")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    return parser.parse_args()

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    # Model Parameters
    input_size = 8
    output_size = 9
    hidden_size = 64
    num_layers = 1
    
    # Dynamic settings based on architecture
    if "Bi-GRU" in args.model:
        dropout = 0.0
        patch_size = 150
    elif "Bi-LSTM" in args.model:
        dropout = 0.1
        patch_size = 150
    elif "Res-ASPP-UNet" in args.model:
        dropout = 0.1 if "LTN" in args.model else 0.0
        patch_size = 160
        
    stride = 100

    # Instantiate Model
    if "Bi-GRU" in args.model:
        model = BiGRU(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    elif "Bi-LSTM" in args.model:
        model = BiLSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    elif "Res-ASPP-UNet" in args.model:
        model = ResASPPUnet(input_features=input_size, output_classes=output_size, dropout_rate=dropout).to(device)

    # Load Weights
    print(f"Loading weights from: {args.weights}")
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # --- INJECT YOUR BLIND DATA HERE ---
    # NOTE: Replace the placeholders below with your actual data variables
    
    # x_global_tensor = torch.cat([x_blind1_tensor, x_blind2_tensor, x_blind3_tensor], dim=0).to(device)
    # y_global_actual = np.concatenate([y_actual_blind1, y_actual_blind2, y_actual_blind3], axis=0)
    
    # To generalize the w1, w2, w3 splitting, just provide the lengths in order:
    # patch_lengths = [len(x_blind1_tensor), len(x_blind2_tensor), len(x_blind3_tensor)]
    # orig_lengths = [original_length1, original_length2, original_length3]
    
    # df_noise = run_global_noise_test(
    #     model=model, 
    #     model_name=args.model, 
    #     x_global_tensor=x_global_tensor, 
    #     y_global_actual=y_global_actual, 
    #     patch_lengths=patch_lengths,
    #     orig_lengths=orig_lengths,
    #     noise_levels=noise_levels,
    #     patch_size=patch_size,
    #     stride=stride
    # )
    
    # Save Results
    # csv_filename = f"{args.model}_global_noise_results.csv"
    # df_noise.to_csv(csv_filename, index=False)
    # print(f"Results stored in '{csv_filename}'.")