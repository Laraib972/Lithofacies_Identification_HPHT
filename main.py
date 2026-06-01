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

# Import your custom modules
# from models import BiLSTM, BiGRU, ResASPPUnet
# from utils import LithoDatasetTrain, LithoDatasetVal
from train_module import train # Importing the engine we just separated

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
    parser = argparse.ArgumentParser(description="Lithofacies Classification Orchestrator")
    
    # Model Selection
    parser.add_argument("--model", type=str, required=True, 
                        choices=["Bi-GRU", "Bi-LSTM", "Res-ASPP-UNet"],
                        help="Select the base architecture to train.")
    
    # LTN Toggle
    parser.add_argument("--use_ltn", action="store_true", 
                        help="Flag to enable LTN. Automatically configures LTN hyperparameters.")
    
    return parser.parse_args()

# ======================================
# 3. Main Execution Block
# ======================================
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    # --- Hyperparameter Dictionary Mapping ---
    # Global settings
    learning_rate = 0.01
    batch_size = 64
    hidden_size = 64
    num_layers = 1
    input_size = 8
    output_size = 9

    # Dynamic table lookup based on model choice and LTN flag
    if args.model == "Bi-GRU":
        epochs = 65 if args.use_ltn else 70
        dropout = 0.0
    elif args.model == "Bi-LSTM":
        epochs = 75 if args.use_ltn else 50
        dropout = 0.1
    elif args.model == "Res-ASPP-UNet":
        epochs = 80 if args.use_ltn else 70
        dropout = 0.1 if args.use_ltn else 0.0

    lambda_ltn = 0.01 if args.use_ltn else 0.0
    model_display_name = f"{args.model}-LTN" if args.use_ltn else args.model
    
    print(f"--- Training {model_display_name} on {device} ---")
    print(f"Parameters: LR={learning_rate}, Epochs={epochs}, Batch={batch_size}, Drop={dropout}, Lambda={lambda_ltn}")

    # --- Setup Data and Rules ---
    with open("rules.json", "r") as f:
        rules = json.load(f)

    class_names = [
        "Marine_Claystone/Shale", "Carbonate", "Carb_Shale",
        "Marine_Sandstone/Shaly_Sand", "Marine_Claystone/Shale_KT",
        "Transitional_Claystone/Shale", "Fluvial_Claystone/Shale",
        "Fluvial_Shaly_Sand", "Fluvial_Sandstone"
    ]

    # Note: Replace the dummy variables below with your actual loaded tensors
    # train_dataset = LithoDatasetTrain(x_train_tensor, y_train_tensor, env_train_encoded)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataset = LithoDatasetVal(x_val_tensor, y_val_tensor)
    # val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- Instantiate Selected Model ---
    if args.model == "Bi-GRU":
        model = BiGRU(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    elif args.model == "Bi-LSTM":
        model = BiLSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    elif args.model == "Res-ASPP-UNet":
        model = ResASPPUnet(input_features=input_size, output_classes=output_size, dropout_rate=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- Execute Training Loop from Module ---
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

    # --- Plot Results ---
    plt.figure(figsize=(12,5))
    plt.suptitle(f"Training Results: {model_display_name}")
    
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")
    
    plot_filename = f"{model_display_name}_training_results.png"
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")

if __name__ == "__main__":
    main()