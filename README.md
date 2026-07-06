# Neurosymbolic AI for Automated Lithofacies Identification

This repository provides a PyTorch implementation of a neurosymbolic framework for automated lithofacies identification. The framework combines deep-learning architectures with optional Logic Tensor Network (LTN) constraints based on depositional/environmental knowledge.

The currently supported architectures are:

- **Bi-GRU**
- **Bi-LSTM**
- **1D Res-ASPP-U-Net**

Each architecture can be trained either:

- as a conventional deep-learning baseline; or
- with the LTN-based logical constraint term enabled.

> **Important data note:** The original well-log datasets used in the associated research are confidential and cannot be distributed. The `.pt` datasets included with this repository are **dummy/example datasets provided only to demonstrate that the code pipeline runs correctly**. They must not be interpreted as the original research data or used to reproduce the numerical results reported in the manuscript.

> **Hyperparameter note:** The experiment-specific hyperparameters and optimization settings used for the reported research results are documented in the associated manuscript. Values present in the scripts and dummy demonstration workflow are intended to make the implementation executable and transparent; users seeking exact experimental reproduction should refer to the manuscript.

---

## Repository Structure

```text
.
├── main.py
├── evaluate.py
├── models.py
├── train_module.py
├── utils.py
├── ltn_module.py
├── rules.json
├── lithofacies_dataset.pt          # Dummy training/validation dataset
├── blind_well1_dataset.pt          # Dummy blind-well dataset
├── saved_models/                   # Created automatically after training
├── Analysis/
│   └── uncertainty_plot.py         # MCDO uncertainty and predictive-entropy visualization
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `main.py` | Main training entry point; selects architecture, enables/disables LTN training, loads the demonstration dataset, trains the model, saves weights, and plots training history |
| `evaluate.py` | Loads trained weights and the dummy blind-well dataset, performs inference, reconstructs the continuous prediction sequence, computes metrics, and saves a confusion matrix |
| `models.py` | Defines ANN, Bi-LSTM, Bi-GRU, and 1D Res-ASPP-U-Net architectures |
| `train_module.py` | Implements model training and validation loops |
| `utils.py` | Contains PyTorch dataset wrappers, Dual Focal Loss, and the combined focal + LTN objective |
| `ltn_module.py` | Implements Łukasiewicz fuzzy-logic operators, aggregation, and canonical LTN loss |
| `rules.json` | Defines environment-dependent allowed lithofacies mappings |
| `lithofacies_dataset.pt` | Dummy serialized tensors for testing the training pipeline |
| `blind_well1_dataset.pt` | Dummy serialized blind-well tensors for testing the evaluation pipeline |
| `Analysis/uncertainty_plot.py` | Performs deterministic and Monte Carlo Dropout inference, reconstructs continuous blind-well outputs, estimates epistemic uncertainty and predictive entropy, and saves an uncertainty-track plot |

---

## Purpose of the Included Dummy Data

The included datasets exist so that a user can clone the repository and verify the complete software workflow without access to the confidential research data.

They are intended to demonstrate:

1. dataset loading;
2. tensor compatibility;
3. model initialization;
4. baseline training;
5. LTN-enhanced training;
6. model-weight saving;
7. blind-well inference;
8. patch reconstruction;
9. metric calculation; and
10. confusion-matrix generation.

The dummy datasets are **not** intended for:

- reproduction of manuscript accuracy or F1 scores;
- geological interpretation;
- comparison with the manuscript's reported model ranking;
- reproduction of the original blind-well experiments; or
- inference about the confidential field data.

Consequently, metrics obtained from the included dummy data are only software-execution checks.

---

## Framework Overview

For conventional baseline training, the optimization objective is the data-driven classification loss:

$$\mathcal{L}_{\mathrm{base}} = \mathcal{L}_{\mathrm{focal}}$$

When LTN training is enabled, the total objective becomes:

$$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{focal}} + \lambda_{\mathrm{LTN}} \mathcal{L}_{\mathrm{LTN}}$$

The LTN component introduces environment-dependent logical constraints. The implemented logical structure penalizes predictions of lithofacies that are disallowed within a specified environment.

Conceptually, the axioms take the form:

$$\forall x, \quad \mathrm{isZone}_j(x) \rightarrow \neg \mathrm{isLithofacies}_i(x)$$

where lithofacies $i$ is disallowed in environment $j$.

### Training and Inference Separation

The repository deliberately separates logical supervision during training from blind-well inference:

- **Baseline training:** data-driven loss only.
- **LTN-enhanced training:** data-driven loss plus LTN penalty.
- **Validation:** performed without applying the LTN penalty.
- **Blind-well inference:** requires only input well-log features and trained model weights.
- **Environment labels are not required for blind-well inference.**

In `evaluate.py`, the `--use_ltn` flag indicates that the loaded weights came from an LTN-trained experiment. It does not calculate an LTN loss during prediction.

---

## Environmental Constraint Rules

The example logical constraints are stored in:

```text
rules.json
```

The current mapping contains four environment IDs:

| Environment ID | Allowed lithofacies |
|---|---|
| `0` | Marine_Claystone/Shale, Carbonate, Marine_Sandstone/Shaly_Sand, Carb_Shale |
| `1` | Marine_Claystone/Shale_KT, Marine_Sandstone/Shaly_Sand, Carb_Shale, Carbonate |
| `2` | Transitional_Claystone/Shale, Marine_Sandstone/Shaly_Sand, Carbonate, Carb_Shale |
| `3` | Carb_Shale, Fluvial_Sandstone, Fluvial_Shaly_Sand, Fluvial_Claystone/Shale |

These rules demonstrate the expected format of the knowledge base.

> Users applying the framework to another basin, stratigraphic setting, facies scheme, or dataset should modify `rules.json` according to their own domain knowledge. Class-name strings must remain consistent with the class names used by the training code.

---

## Dummy Dataset Format

### Training and Validation Dataset

The demonstration training file is:

```text
lithofacies_dataset.pt
```

It is a PyTorch dictionary containing:

```python
{
    "x_train": ...,
    "y_train": ...,
    "x_val": ...,
    "y_val": ...,
    "env_train": ...
}
```

The included dummy tensors have the following shapes:

```text
x_train   : (143, 160, 8)
y_train   : (143, 160, 9)
x_val     : (15, 160, 8)
y_val     : (15, 160, 9)
env_train : (143, 160, 1)
```

General conventions are:

```text
X = (number_of_windows, sequence_length, number_of_input_features)

Y = (number_of_windows, sequence_length, number_of_classes)

ENV = (number_of_windows, sequence_length, 1)
```

The current implementation expects:

```text
Input features : 8
Output classes : 9
```

The target tensors used during training and validation are one-hot encoded.

### Dummy Blind-Well Dataset

The demonstration blind-well file is:

```text
blind_well1_dataset.pt
```

It contains patched blind-well inputs and a continuous true-label sequence for testing reconstruction and evaluation.

The included dummy data use:

```text
x_blind : (41, 160, 8)
y_blind : (6560,)
```

The file also contains reconstruction metadata associated with the dummy patching workflow.

The expected interpretation is:

```text
x_blind =
(number_of_patches, patch_length, number_of_input_features)

y_blind =
(continuous_sequence_length,)
```

> These dimensions describe only the included dummy example. They do not disclose or represent the dimensions of the confidential research dataset.

---

## Patch-Based Input and Reconstruction

The demonstration workflow processes sequential well-log data in windows.

The current dummy blind-well patches follow the same general tail-append logic used during data preparation:

```python
for i in range(0, len(X) - window_size + 1, stride):
    X_windows.append(X[i:i + window_size])

X_windows.append(X[-window_size:])
```

During evaluation:

- predictions are generated patch by patch;
- regular patches are placed back into their corresponding sequence positions;
- the final appended tail patch is aligned with the end of the continuous sequence;
- softmax probabilities are averaged where patch coverage overlaps;
- the final continuous class sequence is obtained with `argmax`.

This reconstruction procedure allows the dummy blind-well workflow to test end-to-end sequence prediction.

> If users replace the dummy data with their own datasets, patch creation and reconstruction geometry must remain consistent. In particular, the window length, stride, tail handling, and sequence length used during preprocessing must match the assumptions used during reconstruction.

---

## Installation

Install the required Python packages:

```bash
pip install torch numpy matplotlib scikit-learn seaborn
```

A CUDA-compatible PyTorch installation may be used for GPU acceleration.

---

## Quick Start

The following commands are intended to let users verify that the implementation runs with the included dummy datasets.

### 1. Train a Base Bi-GRU

```bash
python main.py --model Bi-GRU
```

After successful training, the script creates:

```text
saved_models/Bi-GRU_weights.pth
```

### 2. Evaluate the Trained Base Bi-GRU

```bash
python evaluate.py --model Bi-GRU --weights "saved_models/Bi-GRU_weights.pth" --blind_data "blind_well1_dataset.pt"
```

### 3. Train an LTN-Enhanced Bi-GRU

```bash
python main.py --model Bi-GRU --use_ltn
```

Expected saved weights:

```text
saved_models/Bi-GRU-LTN_weights.pth
```

### 4. Evaluate the LTN-Trained Bi-GRU

```bash
python evaluate.py --model Bi-GRU --use_ltn --weights "saved_models/Bi-GRU-LTN_weights.pth" --blind_data "blind_well1_dataset.pt"
```

The same workflow can be used for the other supported architectures.

---

## Training Commands

### Bi-GRU

Baseline:

```bash
python main.py --model Bi-GRU
```

LTN-enhanced:

```bash
python main.py --model Bi-GRU --use_ltn
```

### Bi-LSTM

Baseline:

```bash
python main.py --model Bi-LSTM
```

LTN-enhanced:

```bash
python main.py --model Bi-LSTM --use_ltn
```

### 1D Res-ASPP-U-Net

Baseline:

```bash
python main.py --model Res-ASPP-UNet
```

LTN-enhanced:

```bash
python main.py --model Res-ASPP-UNet --use_ltn
```

---

## Model Weights

After training finishes, `main.py` automatically creates:

```text
saved_models/
```

and stores the trained model parameters.

Examples include:

```text
saved_models/Bi-GRU_weights.pth
saved_models/Bi-GRU-LTN_weights.pth
saved_models/Bi-LSTM_weights.pth
saved_models/Bi-LSTM-LTN_weights.pth
saved_models/Res-ASPP-UNet_weights.pth
saved_models/Res-ASPP-UNet-LTN_weights.pth
```

The saved files contain:

```python
model.state_dict()
```

These weights can subsequently be loaded by `evaluate.py`.

> The current training script saves the model state produced at the end of training. Users may extend the implementation with validation-based checkpointing if required for their own experiments.

---

## Blind-Well Evaluation

`evaluate.py` is a separate inference script. It does not train the model and does not call `main.py`.

The workflow is:

```text
Train model
    ↓
Save weights
    ↓
Run evaluate.py
    ↓
Load dummy blind-well patches
    ↓
Generate patch-wise logits
    ↓
Apply softmax
    ↓
Reconstruct continuous predictions
    ↓
Calculate evaluation metrics
    ↓
Save confusion matrix
```

### Base Bi-GRU Example

```bash
python evaluate.py --model Bi-GRU --weights "saved_models/Bi-GRU_weights.pth" --blind_data "blind_well1_dataset.pt"
```

### LTN-Trained Bi-GRU Example

```bash
python evaluate.py --model Bi-GRU --use_ltn --weights "saved_models/Bi-GRU-LTN_weights.pth" --blind_data "blind_well1_dataset.pt"
```

The evaluation script reports:

- reconstructed prediction shape;
- accuracy;
- macro F1-score;
- classification report; and
- confusion matrix.

Because the supplied blind-well file is dummy data, these metric values are demonstration outputs only.

---

## Lithofacies Class Ordering

The current implementation uses nine output classes:

```text
0  Marine_Claystone/Shale
1  Carbonate
2  Carb_Shale
3  Marine_Sandstone/Shaly_Sand
4  Marine_Claystone/Shale_KT
5  Transitional_Claystone/Shale
6  Fluvial_Claystone/Shale
7  Fluvial_Shaly_Sand
8  Fluvial_Sandstone
```

This ordering must remain consistent across:

- model output channels;
- one-hot encoded targets;
- blind-well integer labels;
- `class_names`;
- `rules.json`.

Users adapting the code to a different facies scheme must update all relevant components consistently.

---

## Hyperparameters and Experimental Reproduction

The repository is intended to provide the implementation and an executable dummy-data demonstration.

The **authoritative hyperparameter configuration for the research experiments is reported in the associated manuscript**. This includes the model-specific experimental settings selected for the reported comparisons.

Therefore:

- use the included dummy data to verify code execution;
- use the manuscript for the experimental hyperparameters associated with the published results;

Users adapting the framework to new data should independently tune hyperparameters using an appropriate validation strategy.

---

## Reproducibility

The training entry point configures random seeds for:

- Python;
- NumPy;
- PyTorch; and
- CUDA, when available.

It also enables deterministic cuDNN behavior where configured by the script.

Nevertheless, exact numerical reproducibility may depend on:

- hardware;
- operating system;
- PyTorch version;
- CUDA version;
- cuDNN version; and
- nondeterministic backend operations.

---

## Confidential Data and Data Availability

The original well-log datasets used in the research are confidential and cannot be released publicly.

To keep the repository executable, synthetic/dummy serialized datasets are supplied solely for software demonstration and pipeline verification.

Accordingly:

- the dummy data are not the original research data;
- dummy-data metrics are not manuscript results;
- the included examples should not be used for geological interpretation;
- exact experimental hyperparameters should be taken from the manuscript; and
- users wishing to apply the framework must prepare their own data in the documented tensor format.

---

## Analysis Section

Post-training diagnostic workflows are stored separately from the core training and evaluation pipeline:

```text
Analysis/
└── uncertainty_plot.py
```

These analysis scripts use previously trained model weights and do **not** retrain the model.

### Uncertainty Analysis with Monte Carlo Dropout

`Analysis/uncertainty_plot.py` compares deterministic inference with Monte Carlo Dropout (MCDO) inference on a blind-well dataset. It produces continuous tracks for:

- actual lithofacies labels;
- standard deterministic predictions;
- MCDO predictions;
- epistemic uncertainty; and
- predictive entropy.

The workflow is:

```text
Load trained model weights
    ↓
Load patched blind-well data
    ↓
Run standard inference with model.eval()
    ↓
Reactivate dropout modules for MCDO
    ↓
Perform repeated stochastic forward passes
    ↓
Calculate mean predictive probabilities
    ↓
Estimate predictive entropy and epistemic uncertainty
    ↓
Reconstruct patch-wise results to continuous well length
    ↓
Save the uncertainty-track figure
```

The implemented epistemic uncertainty is:

\[
U_{\mathrm{epi}}
=
H\left[\mathbb{E}_{t}(p_t)\right]
-
\mathbb{E}_{t}\left[H(p_t)\right]
\]

where \(p_t\) denotes the predictive probability vector from stochastic forward pass \(t\).

> **Important MCDO requirement:** The selected architecture must contain an actual dropout module in its forward path. The script places the complete model in evaluation mode and then selectively reactivates dropout modules during stochastic inference.

> **Checkpoint consistency:** The model instantiated during analysis must remain compatible with the saved checkpoint. The `--dropout` value should correspond to the intended experimental configuration. Changing dropout only at analysis time changes the stochastic inference procedure.

### `uncertainty_plot.py` Command-Line Arguments

| Argument | Purpose |
|---|---|
| `--model` | Selects the model/checkpoint identifier |
| `--weights` | Path to the saved `.pth` model weights |
| `--blind_data` | Path to the blind-well `.pt` dataset |
| `--well_name` | Name used when constructing the output filename |
| `--passes` | Number of stochastic MCDO forward passes |
| `--plot_start` | First continuous sample index shown in the plot |
| `--plot_end` | End sample index of the plotted interval |
| `--dropout` | Dropout probability used to instantiate the model |
| `--output_dir` | Directory for generated uncertainty plots |

Supported model identifiers are:

```text
Bi-GRU
Bi-GRU-LTN
Bi-LSTM
Bi-LSTM-LTN
Res-ASPP-UNet
Res-ASPP-UNet-LTN
```

### How to Execute `uncertainty_plot.py`

Run commands from the **repository root directory**.

#### Bi-GRU-LTN with dropout 0.1 and 100 MCDO passes

```bash
python Analysis/uncertainty_plot.py --model Bi-GRU-LTN --weights "saved_models/Bi-GRU-LTN_weights.pth" --blind_data "blind_well1_dataset.pt" --passes 100 --dropout 0.1
```

#### Same experiment with an explicit well name

```bash
python Analysis/uncertainty_plot.py --model Bi-GRU-LTN --weights "saved_models/Bi-GRU-LTN_weights.pth" --blind_data "blind_well1_dataset.pt" --well_name "Blind Well 1" --passes 100 --dropout 0.1
```

#### Plot only samples 0 to 3000

```bash
python Analysis/uncertainty_plot.py --model Bi-GRU-LTN --weights "saved_models/Bi-GRU-LTN_weights.pth" --blind_data "blind_well1_dataset.pt" --well_name "Blind Well 1" --passes 100 --dropout 0.1 --plot_start 0 --plot_end 3000
```

#### Use a custom output directory

```bash
python Analysis/uncertainty_plot.py --model Bi-GRU-LTN --weights "saved_models/Bi-GRU-LTN_weights.pth" --blind_data "blind_well1_dataset.pt" --passes 100 --dropout 0.1 --output_dir "Analysis/Uncertainty_Plots"
```

By default, output is written under:

```text
Analysis/Uncertainty_Plots/
```

with a filename of the form:

```text
Bi-GRU-LTN_Blind_Well_Uncertainty_Log.png
```

### Additional Execution Examples

Bi-LSTM:

```bash
python Analysis/uncertainty_plot.py --model Bi-LSTM --weights "saved_models/Bi-LSTM_weights.pth" --blind_data "blind_well1_dataset.pt" --passes 100 --dropout 0.1
```

Bi-LSTM-LTN:

```bash
python Analysis/uncertainty_plot.py --model Bi-LSTM-LTN --weights "saved_models/Bi-LSTM-LTN_weights.pth" --blind_data "blind_well1_dataset.pt" --passes 100 --dropout 0.1
```

Res-ASPP-UNet-LTN:

```bash
python Analysis/uncertainty_plot.py --model Res-ASPP-UNet-LTN --weights "saved_models/Res-ASPP-UNet-LTN_weights.pth" --blind_data "blind_well1_dataset.pt" --passes 100 --dropout 0.1
```

### Dummy-Data Limitation for Uncertainty Analysis

The supplied blind-well dataset is dummy/example data intended only to verify that the analysis pipeline executes. Uncertainty magnitudes and track patterns generated from the dummy data must not be interpreted as manuscript results or geological findings.

For research reproduction, users must use the appropriate trained checkpoint, corresponding research blind-well dataset, consistent patch/reconstruction metadata, and the experiment-specific settings reported in the manuscript.


---

## Citation

If you use this implementation in academic work, please cite the associated manuscript once its final bibliographic information is available.

```bibtex
@article{your_citation_key,
  title   = {Your article title},
  author  = {Author names},
  journal = {Journal name},
  year    = {Year}
}
```

---

## License

Add the appropriate software license before public release.
