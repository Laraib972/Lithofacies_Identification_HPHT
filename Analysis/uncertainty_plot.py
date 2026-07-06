from pathlib import Path
import sys

# ============================================================
# 0. Allow Analysis/ scripts to import repository modules
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# 1. Imports
# ============================================================
import argparse
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models import BiGRU, BiLSTM, ResASPPUnet


# ============================================================
# 2. Reproducibility
# ============================================================
def set_seed(seed=42):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# 3. Load Blind Dataset
# ============================================================
def load_blind_dataset(path):

    print(f"\nLoading blind dataset from:")
    print(path)

    d = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    # --------------------------------------------------------
    # Required keys
    # --------------------------------------------------------
    if "x_blind" not in d:
        raise KeyError(
            "Blind dataset does not contain key 'x_blind'"
        )

    if "y_blind" not in d:
        raise KeyError(
            "Blind dataset does not contain key 'y_blind'"
        )

    x = d["x_blind"].float()
    y = d["y_blind"]

    # --------------------------------------------------------
    # Convert labels to NumPy
    # --------------------------------------------------------
    if torch.is_tensor(y):
        y = y.cpu().numpy()

    y = np.asarray(y).reshape(-1).astype(np.int64)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------
    original_length = int(
        d.get("original_length", len(y))
    )

    patch_size = int(
        d.get("patch_size", x.shape[1])
    )

    stride = int(
        d.get("stride", patch_size)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    if x.ndim != 3:
        raise ValueError(
            "x_blind must have shape "
            "(num_patches, patch_size, num_features). "
            f"Received: {tuple(x.shape)}"
        )

    if x.shape[1] != patch_size:
        raise ValueError(
            f"x_blind patch dimension = {x.shape[1]}, "
            f"but metadata patch_size = {patch_size}"
        )

    if len(y) != original_length:
        raise ValueError(
            f"y_blind length = {len(y)}, "
            f"but original_length = {original_length}"
        )

    print("\nBlind dataset information")
    print("-------------------------")
    print(f"X shape          : {tuple(x.shape)}")
    print(f"Y shape          : {tuple(y.shape)}")
    print(f"Original length  : {original_length}")
    print(f"Patch size       : {patch_size}")
    print(f"Stride           : {stride}")

    return ( x,   y,  original_length,   patch_size,   stride )


# ============================================================
# 4. Reconstruct Continuous Well
# ============================================================
def stitch_tail_append( data, target_length,  patch_size,  stride):

    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    a = np.asarray(data)

    # --------------------------------------------------------
    # Scalar-per-sample patch data:
    # (N, P) -> (N, P, 1)
    # --------------------------------------------------------
    squeezed = (a.ndim == 2)

    if squeezed:
        a = a[..., None]

    if a.ndim != 3:
        raise ValueError(
            "Patched data must have shape "
            "(num_patches, patch_size, channels)"
        )

    num_patches = a.shape[0]
    actual_patch_size = a.shape[1]
    num_channels = a.shape[2]

    if actual_patch_size != patch_size:
        raise ValueError(
            f"Patch dimension = {actual_patch_size}, "
            f"expected patch_size = {patch_size}"
        )

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------
    summed = np.zeros(
        (target_length, num_channels),
        dtype=np.float64
    )

    counts = np.zeros(
        (target_length, 1),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Reconstruct
    #
    # Assumes patch creation:
    # regular windows + final X[-patch_size:] append
    # --------------------------------------------------------
    for i in range(num_patches):

        if i == num_patches - 1:

            # Final tail-appended patch
            start = target_length - patch_size

        else:

            # Regular patch
            start = i * stride

        end = min(
            start + patch_size,
            target_length
        )

        if start < 0:
            raise ValueError(
                "target_length is smaller than patch_size"
            )

        if start >= target_length:
            continue

        length = end - start

        summed[start:end] += a[i, :length]
        counts[start:end] += 1

    # --------------------------------------------------------
    # Coverage validation
    # --------------------------------------------------------
    uncovered = counts[:, 0] == 0

    if np.any(uncovered):

        missing = int(
            uncovered.sum()
        )

        raise ValueError(
            f"Reconstruction left {missing} "
            "uncovered samples. "
            "Check patch_size and stride metadata."
        )

    reconstructed = summed / counts

    if squeezed:
        return reconstructed[:, 0]

    return reconstructed


# ============================================================
# 5. Enable Monte Carlo Dropout
# ============================================================
def enable_mc_dropout(model):

    # Keep BatchNorm and other modules in evaluation mode
    model.eval()
    active_dropout_layers = 0
    print("\nDropout modules detected")
    print("------------------------")
    for name, module in model.named_modules():

        if isinstance(
            module,
            torch.nn.modules.dropout._DropoutNd
        ):

            print(
                f"{name}: "
                f"{module.__class__.__name__}, "
                f"p={module.p}"
            )

            # Only activate genuinely stochastic dropout
            if module.p > 0.0:

                module.train()
                active_dropout_layers += 1

    print(
        f"\nActive MCDO dropout layers: "
        f"{active_dropout_layers}"
    )

    return active_dropout_layers


# ============================================================
# 6. Verify Stochastic Inference
# ============================================================
def verify_mc_stochasticity(  model,  x,   tolerance=1e-12):

    # Use a small subset for verification
    x_test = x[:min(2, len(x))]

    with torch.no_grad():

        output_1 = model(x_test)
        output_2 = model(x_test)

    max_difference = (
        output_1 - output_2
    ).abs().max().item()

    mean_difference = (
        output_1 - output_2
    ).abs().mean().item()

    print("\nMCDO stochasticity check")
    print("------------------------")
    print(
        f"Maximum output difference : "
        f"{max_difference:.10e}"
    )
    print(
        f"Mean output difference    : "
        f"{mean_difference:.10e}"
    )

    if max_difference <= tolerance:

        raise RuntimeError(
            "Repeated MCDO forward passes are "
            "effectively identical. "
            "Dropout is not producing stochastic inference."
        )

    print(
        "Stochastic inference verified successfully."
    )


# ============================================================
# 7. Build Model
# ============================================================
def build_model( name, dropout, device):

    if "Bi-GRU" in name:

        model = BiGRU(
            8,          # input size
            64,         # hidden size
            9,          # output classes
            1,          # recurrent layers
            dropout        )

    elif "Bi-LSTM" in name:

        model = BiLSTM(
            8,
            64,
            9,
            1,
            dropout        )

    else:

        model = ResASPPUnet(
            input_features=8,
            output_classes=9,
            dropout_rate=dropout
        )

    return model.to(device)


# ============================================================
# 8. Plot Tracks
# ============================================================
def plot_tracks( true_labels, standard_prediction, mc_prediction,  epistemic_uncertainty,  predictive_entropy,  model_name,  well_name,  start,  end,  output_dir):

    start = max(0, start)
    end = min(len(true_labels), end)

    if start >= end:

        raise ValueError(
            f"Invalid plot range [{start}, {end}) "
            f"for well length {len(true_labels)}"
        )

    depth_index = np.arange(
        start,
        end
    )

    values = [
        true_labels[start:end],
        standard_prediction[start:end],
        mc_prediction[start:end],
        epistemic_uncertainty[start:end],
        predictive_entropy[start:end]
    ]

    titles = [
        "Actual Lithofacies",
        f"{model_name}\nStandard",
        f"{model_name}\nMCDO",
        "Epistemic Uncertainty",
        "Predictive Entropy"
    ]

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 12),
        sharey=True
    )

    for ax, values_i, title in zip(
        axes,
        values,
        titles
    ):

        ax.plot(
            values_i,
            depth_index,
            linewidth=1.2
        )

        ax.set_title(title)

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4
        )

    axes[0].invert_yaxis()
    axes[0].set_ylabel("Sample index")

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_well_name = (
        well_name.replace(" ", "_")
    )

    output_path = output_dir / (
        f"{model_name}_"
        f"{safe_well_name}_"
        "Uncertainty_Log.png"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved uncertainty plot:\n"
        f"{output_path}"
    )


# ============================================================
# 9. Run Uncertainty Analysis
# ============================================================
def run(  model,  x,  y,  original_length,  patch_size,  stride,  model_name,  well_name,  mc_passes,  plot_start,  plot_end,  output_dir):

    # ========================================================
    # A. Standard deterministic inference
    # ========================================================
    print("\nRunning standard deterministic inference...")

    model.eval()

    with torch.no_grad():

        standard_patch_probs = F.softmax(
            model(x),
            dim=-1
        )

    standard_probs = stitch_tail_append(
        standard_patch_probs,
        original_length,
        patch_size,
        stride
    )

    standard_prediction = np.argmax(
        standard_probs,
        axis=-1
    )

    print(
        "Standard deterministic inference complete."
    )

    # ========================================================
    # B. Enable MC Dropout
    # ========================================================
    print("\nEnabling Monte Carlo Dropout...")

    active_dropout_layers = enable_mc_dropout(
        model
    )

    if active_dropout_layers == 0:

        raise RuntimeError(
            "No active nn.Dropout layer with p > 0 "
            "exists in the instantiated model. "
            "MCDO cannot be performed."
        )

    # ========================================================
    # C. Verify stochasticity
    # ========================================================
    verify_mc_stochasticity(
        model,
        x
    )

    # ========================================================
    # D. Monte Carlo forward passes
    # ========================================================
    print(
        f"\nRunning {mc_passes} "
        "Monte Carlo Dropout passes..."
    )

    mc_samples = []

    with torch.no_grad():

        for iteration in range(mc_passes):

            patch_probs = F.softmax(
                model(x),
                dim=-1
            ).cpu()

            mc_samples.append(
                patch_probs
            )

            print(
                f"\rMCDO pass "
                f"{iteration + 1}/{mc_passes}",
                end="",
                flush=True
            )

    print()

    # Shape:
    # (T, num_patches, patch_size, num_classes)
    mc_samples = torch.stack(
        mc_samples,
        dim=0
    )

    print(
        f"MCDO tensor shape: "
        f"{tuple(mc_samples.shape)}"
    )

    # ========================================================
    # E. Mean predictive probability
    # ========================================================
    mean_patch_probs = mc_samples.mean(
        dim=0
    )

    eps = 1e-10

    # ========================================================
    # F. Predictive Entropy
    #
    # H[E[p(y|x,w)]]
    # ========================================================
    predictive_entropy_patch = -(
        mean_patch_probs
        *
        torch.log(
            mean_patch_probs + eps
        )
    ).sum(dim=-1)

    # ========================================================
    # G. Expected Entropy
    #
    # E[H[p(y|x,w)]]
    # ========================================================
    expected_entropy_patch = -(

        mc_samples
        *
        torch.log(
            mc_samples + eps
        )

    ).sum(dim=-1).mean(dim=0)

    # ========================================================
    # H. Epistemic uncertainty
    #
    # Mutual Information:
    #
    # H[E[p]] - E[H[p]]
    # ========================================================
    epistemic_patch = torch.clamp(

        predictive_entropy_patch
        -
        expected_entropy_patch,

        min=0.0
    )

    # ========================================================
    # I. Reconstruct full well
    # ========================================================
    mc_probs = stitch_tail_append(
        mean_patch_probs,
        original_length,
        patch_size,
        stride
    )

    mc_prediction = np.argmax(
        mc_probs,
        axis=-1
    )

    epistemic_uncertainty = stitch_tail_append(
        epistemic_patch,
        original_length,
        patch_size,
        stride
    )

    predictive_entropy = stitch_tail_append(
        predictive_entropy_patch,
        original_length,
        patch_size,
        stride
    )

    # ========================================================
    # J. Final shape verification
    # ========================================================
    print("\nReconstructed output shapes")
    print("---------------------------")
    print(
        f"True labels           : "
        f"{y.shape}"
    )
    print(
        f"Standard prediction   : "
        f"{standard_prediction.shape}"
    )
    print(
        f"MCDO prediction       : "
        f"{mc_prediction.shape}"
    )
    print(
        f"Epistemic uncertainty : "
        f"{epistemic_uncertainty.shape}"
    )
    print(
        f"Predictive entropy    : "
        f"{predictive_entropy.shape}"
    )

    # ========================================================
    # K. Plot
    # ========================================================
    plot_tracks(
        y,
        standard_prediction,
        mc_prediction,
        epistemic_uncertainty,
        predictive_entropy,
        model_name,
        well_name,
        plot_start,
        plot_end,
        output_dir
    )


# ============================================================
# 10. Command-Line Arguments
# ============================================================
def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Blind-well uncertainty analysis "
            "using Monte Carlo Dropout"
        )
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "Bi-GRU-LTN",
            "Bi-LSTM-LTN",
            "Res-ASPP-UNet-LTN",
            "Bi-GRU",
            "Bi-LSTM",
            "Res-ASPP-UNet"
        ]
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to trained model weights"
    )

    parser.add_argument(
        "--blind_data",
        required=True,
        help="Path to blind-well .pt dataset"
    )

    parser.add_argument(
        "--well_name",
        default="Blind Well"
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=100
    )

    parser.add_argument(
        "--plot_start",
        type=int,
        default=0
    )

    parser.add_argument(
        "--plot_end",
        type=int,
        default=3000
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help=(
            "Dropout probability used to instantiate "
            "the checkpoint architecture"
        )
    )

    parser.add_argument(
        "--output_dir",
        default="Analysis/Uncertainty_Plots"
    )

    return parser.parse_args()


# ============================================================
# 11. Main
# ============================================================
if __name__ == "__main__":

    set_seed(42)

    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Resolve dropout
    # --------------------------------------------------------
    if args.dropout is not None:

        dropout = args.dropout

    else:

        if "Bi-GRU" in args.model:
            dropout = 0.0

        elif "Bi-LSTM" in args.model:
            dropout = 0.1

        elif "LTN" in args.model:
            dropout = 0.1

        else:
            dropout = 0.0

    print(f"Model   : {args.model}")
    print(f"Dropout : {dropout}")
    print(f"Passes  : {args.passes}")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------
    model = build_model( args.model,  dropout,   device  )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------
    print( f"\nLoading weights from:\n"   f"{args.weights}"  )
    state_dict = torch.load( args.weights, map_location=device,  weights_only=True )
    model.load_state_dict( state_dict  )
    print( "Model weights loaded successfully." )

    # --------------------------------------------------------
    # Load blind well
    # --------------------------------------------------------
    ( x_blind,y_blind, original_length, patch_size, stride) = load_blind_dataset( args.blind_data )

    # --------------------------------------------------------
    # Run analysis
    # --------------------------------------------------------
    run( model=model,
        x=x_blind.to(device),
        y=y_blind,
        original_length=original_length,
        patch_size=patch_size,
        stride=stride,
        model_name=args.model,
        well_name=args.well_name,
        mc_passes=args.passes,
        plot_start=args.plot_start,
        plot_end=args.plot_end,
        output_dir=Path(args.output_dir)
    )
