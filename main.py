import argparse
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import os
import numpy as np

from torch.utils.data import DataLoader
from models import BiLSTM, BiGRU, ResASPPUnet
from utils import LithoDatasetTrain, LithoDatasetVal
from train_module import train


# ======================================
# 1. Reproducibility
# ======================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================
# 2. Command Line Argument Parser
# ======================================
def parse_args():
    parser = argparse.ArgumentParser( description="Lithofacies Classification Orchestrator" )
    # Model Selection
    parser.add_argument( "--model", type=str,  required=True,    choices=[ "Bi-GRU",  "Bi-LSTM",  "Res-ASPP-UNet" ],  help="Select the base architecture to train." )
    # LTN Toggle
    parser.add_argument(   "--use_ltn",  action="store_true",   help=(  "Flag to enable LTN. "    "Automatically configures LTN hyperparameters."  ) )
    return parser.parse_args()


# ======================================
# 3. Main Execution Block
# ======================================
def main():
    
    set_seed(42)
    
    device = torch.device( "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    args = parse_args()

    # ==================================
    # Global Hyperparameters
    # ==================================
    learning_rate = 0.01
    batch_size = 64
    hidden_size = 64
    num_layers = 1
    input_size = 8
    output_size = 9

    # ==================================
    # Model-Specific Hyperparameters
    # ==================================
    if args.model == "Bi-GRU":

        epochs = (   65 if args.use_ltn else 70  )
        dropout = 0.0

    elif args.model == "Bi-LSTM":

        epochs = (  75 if args.use_ltn else 50  )
        dropout = 0.1

    elif args.model == "Res-ASPP-UNet":

        epochs = ( 80 if args.use_ltn else 70 )
        dropout = ( 0.1 if args.use_ltn else 0.0 )

    # ==================================
    # LTN Configuration
    # ==================================
    lambda_ltn = (  0.01 if args.use_ltn else 0.0 )

    model_display_name = (
        f"{args.model}-LTN"
        if args.use_ltn
        else args.model
    )

    # ==================================
    # Display Training Configuration
    # ==================================
    print(
        f"\n--- Training "
        f"{model_display_name} "
        f"on {device} ---"
    )

    print(
        f"Parameters: "
        f"LR={learning_rate}, "
        f"Epochs={epochs}, "
        f"Batch={batch_size}, "
        f"Drop={dropout}, "
        f"Lambda={lambda_ltn}"
    )

    # ==================================
    # Load LTN Rules
    # ==================================
    with open( "rules.json",  "r"  ) as f:
        rules = json.load(f)

    # ==================================
    # Lithofacies Class Names
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
    # Load Training Dataset
    # ==================================
    dataset_path = ( "lithofacies_dataset.pt" )

    print(
        f"\nLoading dataset from: "
        f"{dataset_path}"
    )

    data = torch.load( dataset_path,  map_location="cpu",  weights_only=False )

    # ----------------------------------
    # Extract tensors
    # ----------------------------------
    x_train_tensor = data[ "x_train" ]

    y_train_tensor = data[  "y_train" ]

    x_val_tensor = data[ "x_val" ]

    y_val_tensor = data[ "y_val" ]

    env_train_tensor = data[  "env_train"   ]

    # ----------------------------------
    # Display tensor shapes
    # ----------------------------------
    print(  "\nDataset Shapes"    )

    print( "X train :",   x_train_tensor.shape    )

    print(  "Y train :",   y_train_tensor.shape    )

    print(    "X val   :",    x_val_tensor.shape    )

    print(  "Y val   :",   y_val_tensor.shape    )

    print(   "ENV     :",   env_train_tensor.shape    )

    # ==================================
    # Create Datasets
    # ==================================
    train_dataset = LithoDatasetTrain(  x_train_tensor,  y_train_tensor,  env_train_tensor  )

    val_dataset = LithoDatasetVal(   x_val_tensor,   y_val_tensor  )

    # ==================================
    # Create DataLoaders
    # ==================================
    train_loader = DataLoader(  train_dataset,   batch_size=batch_size,  shuffle=True  )

    val_loader = DataLoader(  val_dataset,  batch_size=batch_size,  shuffle=False  )

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
    # Optimizer
    # ==================================
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # ==================================
    # Execute Training Loop
    # ==================================
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        rules=rules,
        class_names=class_names,
        device=device,
        num_epochs=epochs,
        lambda_ltn=lambda_ltn
    )

    # ======================================
    # SAVE TRAINED MODEL WEIGHTS
    # ======================================

    # Create directory if it does not exist
    os.makedirs( "saved_models",  exist_ok=True  )

    # Automatically create model-specific
    # filename
    weights_filename = (
        f"{model_display_name}_weights.pth"
    )

    weights_path = os.path.join(
        "saved_models",
        weights_filename
    )

    # Save only model parameters
    torch.save(
        model.state_dict(),
        weights_path
    )

    print(  "\n================================"  )
    print(     "TRAINED WEIGHTS SAVED"    )
    print(    "================================"   )
    print(     f"Model       : "   f"{model_display_name}"   )
    print(  f"Weights file: "
        f"{weights_path}"    )
    print(     "================================"    )

    # ==================================
    # Plot Training Results
    # ==================================
    plt.figure( figsize=(12, 5) )
    plt.suptitle(   f"Training Results: "   f"{model_display_name}"
    )
    # ----------------------------------
    # Loss
    # ----------------------------------
    plt.subplot( 1,   2,    1  )
    plt.plot( history["train_loss"],  label="Train Loss"  )
    plt.plot( history["val_loss"], label="Val Loss"  )
    plt.legend()
    plt.title( "Loss" )

    # ----------------------------------
    # Accuracy
    # ----------------------------------
    plt.subplot( 1,   2,    2 )
    plt.plot(  history["train_acc"],  label="Train Acc" )
    plt.plot(  history["val_acc"], label="Val Acc" )
    plt.legend()
    plt.title( "Accuracy")
    plt.tight_layout( rect=[0, 0, 1, 0.95] )

    # ==================================
    # Save Training Plot
    # ==================================
    plot_filename = (  f"{model_display_name}"
        f"_training_results.png" )

    plt.savefig( plot_filename,  dpi=300,  bbox_inches="tight" )
    print( f"\nSaved training plot to: "  f"{plot_filename}" )
    plt.show()


# ======================================
# 4. Script Entry Point
# ======================================
if __name__ == "__main__":
    main()