import argparse
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your models are accessible from the main directory
# from models import BiGRU, BiLSTM, ResASPPUnet

# ==========================================
# 1. REPRODUCIBILITY & CONFIGURATION
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

long_labels_original = [
    'Carb_Shale',                   # 0
    'Carbonate',                    # 1
    'Fluvial_Claystone/Shale',      # 2
    'Fluvial_Sandstone',            # 3
    'Fluvial_Shaly_Sand',           # 4
    'Marine_Claystone/Shale',       # 5
    'Marine_Claystone/Shale_KT',    # 6
    'Marine_Sandstone/Shaly_Sand',  # 7
    'Transitional_Claystone/Shale'  # 8
]

short_labels_mapping = [
    'LF8', 'LF6', 'LF3', 'LF5', 
    'LF4', 'LF1', 'LF2', 
    'LF9', 'LF7'
]

# ==========================================
# 2. HELPER FUNCTIONS (Exact Logic)
# ==========================================
def reconstruct_predictions_and_probs(patch_probs, target_length, patch_size, stride):
    """
    Stitches the patch probabilities back into the original well log length.
    """
    num_patches = patch_probs.shape[0]
    num_classes = patch_probs.shape[-1]
    
    summed_probs = np.zeros((target_length, num_classes))
    counts = np.zeros((target_length, 1))

    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        patch_slice_end = patch_size
        
        if end_idx > target_length:
            overhang = end_idx - target_length
            end_idx = target_length
            patch_slice_end = patch_size - overhang
            if start_idx >= target_length: break

        summed_probs[start_idx:end_idx] += patch_probs[i, 0:patch_slice_end, :]
        counts[start_idx:end_idx] += 1

    counts[counts == 0] = 1 
    avg_probs = summed_probs / counts
    final_predictions = np.argmax(avg_probs, axis=-1)
    
    return final_predictions, avg_probs

def enable_dropout(m):
    """ Function to force Dropout layers to stay active during inference """
    if type(m) == torch.nn.Dropout:
        m.train() 

def get_mcd_patch_probs(model, x_tensor, mc_iterations=100):
    """
    Performs Monte Carlo Dropout Inference by accumulating probabilities.
    """
    # 1. Set model to eval mode (freezes BatchNorm)
    model.eval()
    
    # 2. Force Dropout layers to 'train' mode (active)
    model.apply(enable_dropout)
    
    accumulated_probs = None
    
    with torch.no_grad():
        for i in range(mc_iterations):
            logits = model(x_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs += probs
    
    # Average the probabilities across iterations
    mean_patch_probs = accumulated_probs / mc_iterations
    return mean_patch_probs

# ==========================================
# 3. CORE ANALYSIS & PLOTTING PIPELINE
# ==========================================
def run_confidence_analysis(model, model_name, wells_data, mc_iterations, patch_size, stride):
    
    print(f"\n--- Starting MCDO Class Confidence Analysis for {model_name} (T={mc_iterations}) ---")
    
    global_probs_list = []
    global_true_list = []
    
    # Process an arbitrary number of wells dynamically
    for idx, well in enumerate(wells_data):
        print(f"Processing Well {idx+1}/{len(wells_data)}...")
        x_tensor = well['x_tensor']
        orig_len = well['length']
        y_true = well['y_true']  # Assumed to be 1D continuous array of class indices
        
        # MCDO Inference
        avg_patch_probs = get_mcd_patch_probs(model, x_tensor, mc_iterations)
        
        # Stitch
        _, avg_probs_continuous = reconstruct_predictions_and_probs(
            avg_patch_probs, orig_len, patch_size, stride
        )
        
        global_probs_list.append(avg_probs_continuous)
        global_true_list.append(y_true)
        
    # Concatenate all wells
    global_probs = np.concatenate(global_probs_list, axis=0)
    global_true = np.concatenate(global_true_list, axis=0)
    global_confidence = np.max(global_probs, axis=1) # Max probability per sample

    # Collect Stats per Class
    data_list = []
    classes = np.unique(global_true)

    for c in classes:
        idx_mask = (global_true == c)
        vals = global_confidence[idx_mask]
        
        data_list.append({
            'original_index': c,
            'short_label': short_labels_mapping[int(c)],
            'mean_conf': np.mean(vals),
            'std_conf': np.std(vals)
        })

    # Sort by Short Label (LF1 -> LF9)
    df_plot = pd.DataFrame(data_list)
    df_plot = df_plot.sort_values('short_label')

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    sns.set_style("ticks") 

    x_pos = np.arange(len(df_plot))

    # Plot Error Bars
    plt.errorbar(
        x_pos, 
        df_plot['mean_conf'], 
        yerr=df_plot['std_conf'], 
        fmt='o',              
        markersize=12,        
        capsize=6,            
        elinewidth=2,         
        markeredgewidth=2,    
        color='navy',         
        ecolor='black',       
        label=f'MCDO Mean Conf. ± Std ({model_name})'
    )

    # Threshold Line
    plt.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='High Confidence (0.9)')

    # Add Value Labels
    for i, (mean_val, std_val) in enumerate(zip(df_plot['mean_conf'], df_plot['std_conf'])):
        text_y_pos = mean_val + std_val + 0.02
        plt.text(i, text_y_pos, f"{mean_val:.2f}", ha='center', fontsize=20)

    # Labels & Ticks
    plt.xlabel("Lithofacies Class", fontsize=25)
    plt.ylabel("Mean Prediction Probability", fontsize=25)
    plt.xticks(x_pos, df_plot['short_label'], rotation=0, ha='center', fontsize=20)
    plt.yticks(fontsize=20)

    # Limits & Grid
    plt.ylim(0, 1.15)
    plt.grid(True, axis='y', linestyle=':', alpha=0.6) 
    plt.legend(loc='lower left', fontsize=20, frameon=True) 

    # Save Dynamic Output
    # Ensure directory exists
    os.makedirs('Class_Confidence', exist_ok=True)
    filename = f'Class_Confidence/Prediction_Confidence_Overall_MCDO_{model_name}.tiff'
    plt.savefig(filename, dpi=600, format='tiff', bbox_inches='tight')
    print(f"✅ Plot successfully saved to {filename}")
    plt.show()

# ==========================================
# 4. COMMAND LINE INTERFACE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Class-wise MCDO Confidence Analysis")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["Bi-GRU-LTN", "Bi-LSTM-LTN", "Res-ASPP-UNet-LTN", "Bi-GRU", "Bi-LSTM", "Res-ASPP-UNet"])
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--passes", type=int, default=100, help="Number of Monte Carlo iterations")
    return parser.parse_args()

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # Enforce standard hyperparams
    input_size = 8
    output_size = 9
    hidden_size = 64
    num_layers = 1
    
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
    # NOTE: Replace with your actual tensors and 1D label arrays
    
    # wells_data = [
    #     {'x_tensor': x_blind1_tensor.to(device), 'length': original_length1, 'y_true': np.argmax(y_blind1_enc, axis=1)},
    #     {'x_tensor': x_blind2_tensor.to(device), 'length': original_length2, 'y_true': np.argmax(y_blind2_enc, axis=1)},
    #     {'x_tensor': x_blind3_tensor.to(device), 'length': original_length3, 'y_true': np.argmax(y_blind3_enc, axis=1)}
    # ]
    
    # run_confidence_analysis(
    #     model=model, 
    #     model_name=args.model, 
    #     wells_data=wells_data, 
    #     mc_iterations=args.passes,
    #     patch_size=patch_size, 
    #     stride=stride
    # )
