import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Assuming your models are accessible from the main directory
# from models import BiGRU, BiLSTM, ResASPPUnet

# ==========================================
# 0. REPRODUCIBILITY SETUP
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
# 1. HELPER: DYNAMIC STITCHING FUNCTION
# ==========================================
def stitch_patches(patched_data, target_length, patch_size, stride):
    """
    Stitches patches back to original length, handling overlaps by averaging.
    Input: (Num_Patches, Patch_Size, Channels) or (Num_Patches, Patch_Size)
    Output: (Target_Length, Channels) or (Target_Length,)
    """
    if torch.is_tensor(patched_data):
        patched_data = patched_data.cpu().numpy()
        
    # Handle 2D input (Uncertainty maps) by adding a channel dim
    squeezed = False
    if patched_data.ndim == 2:
        patched_data = np.expand_dims(patched_data, axis=-1)
        squeezed = True
        
    num_patches = patched_data.shape[0]
    channels = patched_data.shape[-1]
    
    summed_data = np.zeros((target_length, channels))
    counts = np.zeros((target_length, 1))

    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_size
        
        # Handle Edge Cases
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
    counts[counts == 0] = 1 
    avg_data = summed_data / counts
    
    if squeezed:
        return avg_data.flatten()
    return avg_data

# ==========================================
# 2. PLOTTING FUNCTION
# ==========================================
def plot_5_track_log(true, pred_std, pred_mcdo, epistemic_unc, total_unc, 
                     model_name, well_name, start=0, end=500):
    
    # Create depth array (index)
    depth = np.arange(len(true))
    
    # Slice the data for the specific zoom window
    mask = (depth >= start) & (depth < end)
    d_seg = depth[mask]
    t_seg = true[mask]
    p_std_seg = pred_std[mask]
    p_mcdo_seg = pred_mcdo[mask]
    u_epi_seg = epistemic_unc[mask]
    u_tot_seg = total_unc[mask]
    
    # Create 5 Subplots sharing Y axis (Depth)
    fig, ax = plt.subplots(1, 5, figsize=(20, 12), sharey=True)
    
    # --- Track 1: Actual Lithofacies ---
    ax[0].plot(t_seg, d_seg, color='black', lw=1.5)
    ax[0].set_title("Track 1\nActual Lithofacies", fontsize=11, fontweight='bold')
    ax[0].set_xlabel("Class Index")
    ax[0].invert_yaxis() 
    ax[0].grid(True, which='both', linestyle='--', alpha=0.5)
    
    # --- Track 2: Standard Prediction ---
    ax[1].plot(p_std_seg, d_seg, color='blue', lw=1.5)
    ax[1].set_title(f"Track 2\n{model_name}\n(Standard)", fontsize=11, fontweight='bold')
    ax[1].set_xlabel("Class Index")
    ax[1].grid(True, which='both', linestyle='--', alpha=0.5)
    
    # --- Track 3: MCDO Prediction ---
    ax[2].plot(p_mcdo_seg, d_seg, color='green', lw=1.5)
    ax[2].set_title(f"Track 3\n{model_name}\n(MCDO Robust)", fontsize=11, fontweight='bold')
    ax[2].set_xlabel("Class Index")
    ax[2].grid(True, which='both', linestyle='--', alpha=0.5)
    
    # --- Track 4: Epistemic Uncertainty ---
    ax[3].plot(u_epi_seg, d_seg, color='firebrick', lw=1)
    ax[3].fill_betweenx(d_seg, 0, u_epi_seg, color='firebrick', alpha=0.3)
    ax[3].set_title("Track 4\nEpistemic Uncertainty\n(Model Ignorance)", fontsize=11, fontweight='bold')
    ax[3].set_xlabel("Uncertainty (Nats)")
    ax[3].set_xlim(0, max(u_epi_seg.max(), 0.1) * 1.1) 
    ax[3].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Track 5: Total Uncertainty ---
    ax[4].plot(u_tot_seg, d_seg, color='purple', lw=1)
    ax[4].fill_betweenx(d_seg, 0, u_tot_seg, color='purple', alpha=0.3)
    ax[4].set_title("Track 5\nTotal Uncertainty\n(Entropy)", fontsize=11, fontweight='bold')
    ax[4].set_xlabel("Entropy (Nats)")
    ax[4].set_xlim(0, max(u_tot_seg.max(), 0.1) * 1.1)
    ax[4].grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.suptitle(f"{well_name} Analysis: Full Uncertainty Profile ({start} to {end})", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_filename = f"{model_name}_{well_name.replace(' ', '_')}_Uncertainty_Log.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nSaved uncertainty track log to {plot_filename}")
    plt.show()

# ==========================================
# 3. CORE GENERATION PIPELINE
# ==========================================
def generate_and_plot_5track(model, x_tensor, y_actual_continuous, original_length, 
                             patch_size, stride, model_name, well_name, T, start, end):
    
    print(f"\n--- Processing Data for {well_name} using {model_name} ---")
    
    # A. Standard Prediction
    model.eval() 
    with torch.no_grad():
        logits_std = model(x_tensor) 
        probs_std = F.softmax(logits_std, dim=-1)
        
        # Stitch
        probs_std_stitched = stitch_patches(probs_std, original_length, patch_size, stride)
        y_pred_standard = np.argmax(probs_std_stitched, axis=-1) 

    # B. MCDO Prediction & Uncertainty
    model.train() # Force Dropout ON
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            m.requires_grad_(False)

    mc_probs_list = []
    print(f"Running MCDO Forward Passes (T={T})...")
    with torch.no_grad():
        for t in range(T):
            logits = model(x_tensor)
            probs = F.softmax(logits, dim=-1)
            mc_probs_list.append(probs.cpu())

    # Stack: (T, Num_Patches, Patch_Size, Classes)
    mc_stack = torch.stack(mc_probs_list)

    # --- Calculate Stats per Patch ---
    epsilon = 1e-10
    mean_probs_patch = torch.mean(mc_stack, dim=0) 

    # Uncertainties
    total_unc_patch = -torch.sum(mean_probs_patch * torch.log(mean_probs_patch + epsilon), dim=-1)
    ind_entropy = -torch.sum(mc_stack * torch.log(mc_stack + epsilon), dim=-1)
    aleatoric_unc_patch = torch.mean(ind_entropy, dim=0)
    epistemic_unc_patch = total_unc_patch - aleatoric_unc_patch

    # --- STITCH EVERYTHING ---
    print("Stitching probability and uncertainty arrays to continuous well length...")
    
    probs_mcdo_stitched = stitch_patches(mean_probs_patch, original_length, patch_size, stride)
    y_pred_mcdo = np.argmax(probs_mcdo_stitched, axis=-1)
    
    uc_epistemic_stitched = stitch_patches(epistemic_unc_patch, original_length, patch_size, stride)
    uc_total_stitched = stitch_patches(total_unc_patch, original_length, patch_size, stride)

    print("✅ Data Generation Complete. Rendering plot...")
    
    # Execute Plot
    plot_5_track_log(
        true=y_actual_continuous, 
        pred_std=y_pred_standard, 
        pred_mcdo=y_pred_mcdo, 
        epistemic_unc=uc_epistemic_stitched, 
        total_unc=uc_total_stitched,
        model_name=model_name,
        well_name=well_name,
        start=start, 
        end=end
    )

# ==========================================
# 4. COMMAND LINE INTERFACE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate 5-Track Uncertainty Log")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["Bi-GRU-LTN", "Bi-LSTM-LTN", "Res-ASPP-UNet-LTN", "Bi-GRU", "Bi-LSTM", "Res-ASPP-UNet"])
    parser.add_argument("--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--well_name", type=str, default="Blind Well", help="Name of well for plot title")
    parser.add_argument("--passes", type=int, default=100, help="Number of Monte Carlo passes")
    parser.add_argument("--plot_start", type=int, default=0, help="Start depth/index for the plot zoom")
    parser.add_argument("--plot_end", type=int, default=3000, help="End depth/index for the plot zoom")
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
    
    # Dynamic settings based on your table
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
    # NOTE: Replace these placeholders with your actual variables (e.g., x_blind2_tensor, original_length2, etc.)
    
    # x_tensor = x_blind2_tensor.to(device)
    # original_length = original_length2
    
    # Assuming you have the original continuous Y. If not, reconstruct it from the tensor:
    # y_probs_orig = stitch_patches(y_blind2_tensor, original_length, patch_size, stride)
    # y_actual_continuous = np.argmax(y_probs_orig, axis=-1)
    
    # generate_and_plot_5track(
    #     model=model, 
    #     x_tensor=x_tensor, 
    #     y_actual_continuous=y_actual_continuous, 
    #     original_length=original_length, 
    #     patch_size=patch_size, 
    #     stride=stride, 
    #     model_name=args.model,
    #     well_name=args.well_name,
    #     T=args.passes,
    #     start=args.plot_start,
    #     end=args.plot_end
    # )
