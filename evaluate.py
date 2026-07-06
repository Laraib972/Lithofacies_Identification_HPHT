import argparse
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    f1_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from models import BiLSTM, BiGRU, ResASPPUnet


# ======================================
# 1. Reconstruction Logic
# ======================================
def reconstruct_predictions( patch_probs,  target_length,  patch_size,  stride):
   
    num_patches = patch_probs.shape[0]
    num_classes = patch_probs.shape[-1]

    summed_probs = np.zeros(
        (target_length, num_classes),
        dtype=np.float64    )

    counts = np.zeros(
        (target_length, 1),
        dtype=np.float64    )

    # ==================================
    # Regular patches
    # All except final appended tail
    # ==================================
    for i in range(num_patches - 1):

        start_idx = i * stride
        end_idx = start_idx + patch_size

        if end_idx > target_length:
            raise ValueError(
                f"Regular patch {i} exceeds "
                f"well length: "
                f"[{start_idx}:{end_idx}] "
                f"for target length "
                f"{target_length}"            )

        summed_probs[     start_idx:end_idx    ] += patch_probs[i]

        counts[   start_idx:end_idx    ] += 1.0

    # ==================================
    # Final tail patch
    # Exact inverse of X[-patch_size:]
    # ==================================
    tail_start = target_length - patch_size
    tail_end = target_length

    summed_probs[   tail_start:tail_end  ] += patch_probs[-1]

    counts[    tail_start:tail_end   ] += 1.0

    # ==================================
    # Check complete coverage
    # ==================================
    uncovered = np.where(    counts[:, 0] == 0  )[0]

    if len(uncovered) > 0:
        raise ValueError(
            f"{len(uncovered)} samples were "
            f"not covered during reconstruction. "
            f"First uncovered indices: "
            f"{uncovered[:10]}"        )

    # ==================================
    # Average probabilities
    # ==================================
    avg_probs = summed_probs / counts
    # ==================================
    # Final class prediction
    # ==================================
    final_predictions = np.argmax(   avg_probs,    axis=-1  )
    return final_predictions


# ======================================
# 2. Generalized Evaluation Pipeline
# ======================================
def evaluate_blind_well(  model, x_blind_patched,  y_blind_true, original_length,  patch_size,  stride,  class_names,  device,  model_name,  plot_cm=True):
    
    print( f"\n--- Evaluating {model_name} "
        f"on Blind Well "
        f"(Length: {original_length}) ---"    )

    # ==================================
    # Prepare input tensor
    # ==================================
    if not torch.is_tensor(x_blind_patched):

        x_blind_tensor = torch.tensor( x_blind_patched,  dtype=torch.float32  )

    else:

        x_blind_tensor = ( x_blind_patched.float()   )

    x_blind_tensor = ( x_blind_tensor.to(device)   )

    print(  "Input patch shape       :",  tuple(x_blind_tensor.shape)    )

    # ==================================
    # Model inference
    # ==================================
    model.eval()

    with torch.no_grad():

        logits = model( x_blind_tensor )

        probs = F.softmax( logits,  dim=-1  ).cpu().numpy()

    print( "Logits shape            :",  tuple(logits.shape) )
    print( "Probability shape       :",    probs.shape  )

    # ==================================
    # Output-shape validation
    # ==================================
    if probs.ndim != 3:

        raise ValueError(
            f"Expected model probabilities with "
            f"shape "
            f"(N_patches, patch_size, classes), "
            f"got {probs.shape}"
        )

    if probs.shape[0] != x_blind_tensor.shape[0]:

        raise ValueError(
            "Number of output patches does not "
            "match number of input patches."
        )

    if probs.shape[1] != patch_size:

        raise ValueError(
            f"Model output sequence length is "
            f"{probs.shape[1]}, but blind-data "
            f"patch size is {patch_size}."
        )

    # ==================================
    # Reconstruct continuous prediction
    # ==================================
    predicted_labels = reconstruct_predictions(
        patch_probs=probs,
        target_length=original_length,
        patch_size=patch_size,
        stride=stride
    )

    print(
        "Reconstructed shape     :",
        predicted_labels.shape
    )

    # ==================================
    # Prediction length check
    # ==================================
    if predicted_labels.shape[0] != original_length:

        raise ValueError(
            f"Prediction length mismatch. "
            f"Expected {original_length}, "
            f"got {predicted_labels.shape[0]}"
        )

    # ==================================
    # Prepare true labels
    # ==================================
    if torch.is_tensor(y_blind_true):

        true_labels = (
            y_blind_true
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

    else:

        true_labels = (
            np.asarray(y_blind_true)
            .flatten()
        )

    true_labels = true_labels.astype(
        np.int64
    )

    print(
        "True-label shape        :",
        true_labels.shape
    )

    # ==================================
    # True-label length check
    # ==================================
    if true_labels.shape[0] != original_length:

        raise ValueError(
            f"True-label length mismatch. "
            f"Expected {original_length}, "
            f"got {true_labels.shape[0]}"
        )

    # ==================================
    # Class validation
    # ==================================
    if true_labels.size == 0:

        raise ValueError(
            "True-label array is empty."
        )

    if true_labels.min() < 0:

        raise ValueError(
            f"Negative class ID found: "
            f"{true_labels.min()}"
        )

    if true_labels.max() >= len(class_names):

        raise ValueError(
            f"Invalid class ID "
            f"{true_labels.max()}. "
            f"Expected class IDs from "
            f"0 to {len(class_names) - 1}."
        )

    # ==================================
    # Classes present in blind well
    # ==================================
    present_classes = np.unique(
        true_labels
    )

    present_class_names = [
        class_names[int(i)]
        for i in present_classes
    ]

    print(
        "Present class IDs       :",
        present_classes
    )

    # ==================================
    # Metrics
    # ==================================
    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        labels=present_classes,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=present_classes,
        target_names=present_class_names,
        digits=4,
        zero_division=0
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"F1-score (macro): "
        f"{macro_f1:.4f}"
    )

    print(
        f"\nClassification Report:\n"
        f"{report}"
    )

    # ==================================
    # Confusion Matrix
    # ==================================
    if plot_cm:

        cm = confusion_matrix(
            true_labels,
            predicted_labels,
            labels=present_classes
        )

        plt.figure(
            figsize=(10, 8)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=present_class_names,
            yticklabels=present_class_names
        )

        plt.title(
            f"Confusion Matrix: "
            f"{model_name}"
        )

        plt.ylabel(
            "True Lithofacies"
        )

        plt.xlabel(
            "Predicted Lithofacies"
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.yticks(
            rotation=0
        )

        plt.tight_layout()

        # Make filename filesystem-safe
        safe_model_name = (
            model_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        plot_filename = (
            f"{safe_model_name}_blind_cm.png"
        )

        plt.savefig(
            plot_filename,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"\nSaved confusion matrix to: "
            f"{plot_filename}"
        )

        plt.show()

    return (
        predicted_labels,
        accuracy,
        macro_f1
    )


# ======================================
# 3. Command Line Argument Parser
# ======================================
def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Blind Well Evaluation Script"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "Bi-GRU",
            "Bi-LSTM",
            "Res-ASPP-UNet"
        ],
        help=(
            "Select the architecture "
            "to evaluate."
        )
    )

    parser.add_argument(
        "--use_ltn",
        action="store_true",
        help=(
            "Flag indicating that the loaded "
            "weights belong to an LTN-trained "
            "model."
        )
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=(
            "Path to saved model weights."
        )
    )

    parser.add_argument(
        "--blind_data",
        type=str,
        required=True,
        help=(
            "Path to blind-well .pt dataset."
        )
    )

    return parser.parse_args()


# ======================================
# 4. Main Execution Block
# ======================================
def main():

    # ==================================
    # Device
    # ==================================
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args = parse_args()

    # ==================================
    # Shared Model Hyperparameters
    # ==================================
    hidden_size = 64
    num_layers = 1
    input_size = 8
    output_size = 9

    # ==================================
    # Model-Specific Architecture
    #
    # IMPORTANT:
    # patch_size is NOT defined here.
    # It belongs to the blind dataset.
    # ==================================
    if args.model == "Bi-GRU":

        dropout = 0.0

    elif args.model == "Bi-LSTM":

        dropout = 0.1

    elif args.model == "Res-ASPP-UNet":

        dropout = (
            0.1
            if args.use_ltn
            else 0.0
        )

    # ==================================
    # Display Name
    # ==================================
    model_display_name = (
        f"{args.model}-LTN"
        if args.use_ltn
        else args.model
    )

    # ==================================
    # Lithofacies Classes
    # ==================================
    class_names = [
        "Marine_Claystone/Shale",
        "Carbonate",
        "Carb_Shale",
        "Marine_Sandstone/Shaly_Sand",
        "Marine_Claystone/Shale_KT",
        "Transitional_Claystone/Shale",
        "Fluvial_Claystone/Shale",
        "Fluvial_Shaly_Sand",
        "Fluvial_Sandstone"
    ]

    # ==================================
    # Load Blind-Well Dataset
    # ==================================
    print(
        f"\nLoading blind dataset from: "
        f"{args.blind_data}"
    )

    blind_data = torch.load(
        args.blind_data,
        map_location="cpu",
        weights_only=False
    )

    # ==================================
    # Validate expected dictionary keys
    # ==================================
    required_keys = {
        "x_blind",
        "y_blind"
    }

    missing_keys = (
        required_keys
        - set(blind_data.keys())
    )

    if missing_keys:

        raise KeyError(
            f"Blind dataset is missing keys: "
            f"{sorted(missing_keys)}"
        )

    # ==================================
    # Separate X and Y
    # ==================================
    x_blind_patched = blind_data[
        "x_blind"
    ]

    y_blind_true = blind_data[
        "y_blind"
    ]

    # ==================================
    # Validate blind input shape
    # ==================================
    if x_blind_patched.ndim != 3:

        raise ValueError(
            f"Expected x_blind shape "
            f"(N_patches, patch_size, features), "
            f"got "
            f"{tuple(x_blind_patched.shape)}"
        )

    # ==================================
    # Infer geometry directly from data
    # ==================================
    num_patches = (
        x_blind_patched.shape[0]
    )

    patch_size = (
        x_blind_patched.shape[1]
    )

    num_features = (
        x_blind_patched.shape[2]
    )

    original_length = (
        y_blind_true.numel()
        if torch.is_tensor(y_blind_true)
        else np.asarray(y_blind_true).size
    )

    # ==================================
    stride = patch_size

    # ==================================
    # Validate feature count
    # ==================================
    if num_features != input_size:

        raise ValueError(
            f"Blind dataset contains "
            f"{num_features} features, "
            f"but model expects "
            f"{input_size} features."
        )

    # ==================================
    # Display Configuration
    # ==================================
    print(  "\n================================"  )
    print(    "Blind Well Dataset Information"  )
    print(  "================================"  )
    print(  f"Model            : "     f"{model_display_name}"   )
    print(    f"Device           : "     f"{device}"  )
    print(   f"Number patches   : "    f"{num_patches}"  )
    print(  f"Patch size       : "    f"{patch_size}"  )
    print(  f"Stride           : "   f"{stride}"  )
    print(  f"Input features   : "   f"{num_features}"  )
    print( f"Original length  : "   f"{original_length}" )
    print(  "================================" )

    # ==================================
    # Instantiate Selected Model
    # ==================================
    if args.model == "Bi-GRU":

        model = BiGRU(
            input_size,
            hidden_size,
            output_size,
            num_layers,
            dropout
        ).to(device)

    elif args.model == "Bi-LSTM":

        model = BiLSTM(
            input_size,
            hidden_size,
            output_size,
            num_layers,
            dropout
        ).to(device)

    elif args.model == "Res-ASPP-UNet":

        model = ResASPPUnet(
            input_features=input_size,
            output_classes=output_size,
            dropout_rate=dropout
        ).to(device)

    # ==================================
    # Load Trained Weights
    # ==================================
    print(  f"\nLoading weights from: "
        f"{args.weights}"    )

    state_dict = torch.load(   args.weights,   map_location=device,     weights_only=True )

    model.load_state_dict(  state_dict  )
    print( "Model weights loaded successfully."    )

    # ==================================
    # Execute Blind-Well Evaluation
    # ==================================
    predicted_labels, accuracy, macro_f1 = (
        evaluate_blind_well(
            model=model,
            x_blind_patched=x_blind_patched,
            y_blind_true=y_blind_true,
            original_length=original_length,
            patch_size=patch_size,
            stride=stride,
            class_names=class_names,
            device=device,
            model_name=model_display_name,
            plot_cm=True
        )
    )

    # ==================================
    # Final Summary
    # ==================================
    print( "\n================================"  )
    print(  "Evaluation Completed"  )
    print(  "================================"  )
    print(  f"Prediction shape : "    f"{predicted_labels.shape}" )
    print(
        f"Accuracy         : "    f"{accuracy:.4f}"   )
    print(  f"Macro F1         : "    f"{macro_f1:.4f}"  )


if __name__ == "__main__":
    main()