# Neurosymbolic AI for Automated Lithofacies Identification

This repository contains the official PyTorch implementation of a neurosymbolic framework that integrates Deep Learning architectures (Bi-GRU, Bi-LSTM, 1D Res-ASPP-U-Net) with Logic Tensor Networks (LTN). 
This methodology enhances standard data-driven predictions by enforcing environmental constraints using Łukasiewicz logic. The repository also includes a comprehensive uncertainty quantification suite utilizing Monte Carlo Dropout (MCDO) to evaluate model reliability and noise sensitivity analysis.

## 📌 Repository Structure

```text
├── main.py                          # Orchestrator for training the models
├── evaluate.py                      # Inference and evaluation on blind wells 
├── models.py                        # Deep learning architectures (ANN, Bi-LSTM, Bi-GRU, Res-ASPP-U-Net)
├── ltn_module.py                    # Logic Tensor Network operations and canonical loss functions
├── train_module.py                  # Training loop integrating Dual Focal Loss and LTN penalties
├── utils.py                         # Data loading and loss function utilities
├── rules.json                       # Environmental constraints and allowed lithofacies mappings
├── Analysis/
│   ├── noise_sensitivity.py         # Evaluates model degradation under graded Gaussian noise
│   ├── class_confidence_mcdo.py     # Calculates prediction probability and variance per facies class
│   ├── compare_uncertainty_classes.py # Comparative bar charts for epistemic uncertainty across models
│   └── uncertainty_plot.py          # Generates continuous depth epismetic uncertainty plot
└── README.md
```

## 📊 Data Requirements & Formatting

Due to strict Non-Disclosure Agreements (NDA) regarding the well logs utilized in this research, the original training, validation, and blind well datasets cannot be shared publicly.

To execute or reproduce this methodology, users must provide their own well log datasets or generate synthetic dummy arrays. The data must be processed as sequences using a sliding window approach, matching the exact tensor shapes defined below.

### 1. Training Tensor Shapes (`main.py`)
Your training data must be formatted into 3D tensors with the following dimensions:

* **`X_train` (Input Features):** `(Num_Windows, Window_Size, 8)`
  * *Description:* 8 continuous well log curves.
* **`Y_train` (Target Labels):** `(Num_Windows, Window_Size, 9)`
  * *Description:* One-hot encoded ground truth for the 9 target lithofacies classes.
* **`X_env` (Environment Constraints):** `(Num_Windows, Window_Size, 1)`
  * *Description:* Categorical integers (0 to 3) mapped directly to the logic conditions defined in `rules.json`.

### 2. Inference Tensor Shapes (`evaluate.py` & `Analysis/`)
For predicting or analyzing a continuous blind well, only the patched input features and the original continuous depth length are required.

* **`X_blind` (Input Features):** `(Total_Patches, Window_Size, 8)`
* **Evaluation Note:** The inference scripts utilize an overhang clipping logic to reconstruct continuous logs by averaging overlapping patches. Zero-count classes are automatically excluded during evaluation to prevent artificial deflation of the Macro F1-score.

## 🛠️ Usage Instructions

### 1. Training
Train a model using the command-line interface. Use the `--use_ltn` flag to activate the neurosymbolic logical constraints.
```bash
# Example: Train 1D Res-ASPP-U-Net with LTN constraints
python main.py --model Res-ASPP-UNet --use_ltn
```

### 2. Inference (Blind Well Evaluation)
Run inference to generate a classification report and confusion matrix. Overlapping windows are automatically stitched back into a continuous well log.
```bash
# Example: Evaluate the Bi-GRU model
python evaluate.py --model Bi-GRU --use_ltn --weights path/to/weights.pth --stride 100
```

### 3. Uncertainty & Robustness Analysis
Navigate to the `Analysis` folder to execute the MCDO and noise testing suites.

* **Generate continuous depth epistemic uncertainty plot Log:**
```bash
python Analysis/uncertainty_plot.py --model Bi-LSTM --weights path/to/weights.pth --passes 100
```

* **Assess Class-wise Confidence (MCDO):**
```bash
python Analysis/class_confidence_mcdo.py --model Res-ASPP-UNet-LTN --weights path/to/weights.pth --passes 100
```

* **Execute Global Noise Sensitivity Test:**
```bash
python Analysis/noise_sensitivity.py --model Bi-GRU-LTN --weights path/to/weights.pth
```