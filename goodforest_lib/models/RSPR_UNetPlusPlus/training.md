# README for Training Script

This README provides an overview of the functionalities in the training script, including model training, validation, loss calculation, and logging. The script trains a segmentation model, computes metrics, and saves checkpoints for later use.

---

## Script Overview

This script trains a PyTorch-based segmentation model using a combined loss function (Dice loss + Cross-Entropy loss + optional False Positive penalty) and validates the model on a separate validation set. It logs training progress to TensorBoard, tracks performance metrics, and saves the best model based on Mean IoU.

---

## Main Functions and Components

### 1. **Loss Functions**

- **`dice_loss`**: Calculates the Dice loss to measure overlap between predicted and target masks, which helps handle class imbalance.
- **`combined_loss`**: Combines Dice loss, Cross-Entropy loss, and optionally a False Positive penalty to balance class prediction accuracy with segmentation quality.

### 2. **Metric Calculation**

- **`calculate_metrics`**: Computes IoU, F1-Score, Precision, and Recall for each class, and overall accuracy. This helps evaluate segmentation performance on each class independently. (When one refers to metrics in this Readme, those are the ones described)

### 3. **Validation and Logging**

- **`validate`**: Evaluates the model on the validation set and calculates average loss and metrics.
- **`log_validation_results`**: Logs validation loss and metrics to TensorBoard for easy tracking and visualization.
- **`validation_epoch`**: Runs a full validation epoch, including metric calculation and logging.

### 4. **Training Routine**

- **`train`**: The main training function, which includes:
  - Training loop with logging for loss and accuracy.
  - Validation after each epoch, with Mean IoU used to adjust the learning rate (scheduler).
  - Checkpointing the best model and saving current progress in case of interruption.
  - TensorBoard logging for loss, accuracy, learning rate, and other metrics.

---

## Usage

1. **Parameters**
   - **Model**: A PyTorch model for segmentation.
   - **Data Loaders**: `train_loader` for training and `val_loader` for validation.
   - **Hyperparameters**:
     - `num_epochs`: Total number of training epochs.
     - `num_classes`: Number of segmentation classes.
     - `weights`: Tuple containing weights for Dice and Cross-Entropy losses.
   - **Optional**:
     - `checkpoint_dir`: Directory to save model checkpoints.
     - `log_dir`: Directory for TensorBoard logs.
     - `resume_checkpoint`: Path to a checkpoint file to resume training.
     - `data_description` and `model_description`: Descriptions for logging additional info to TensorBoard.

2. **Run Training**
   - Initialize the model and data loaders.
   - Call the `train` function with required arguments.

3. **Output**
   - TensorBoard logs for visualizing training progress.
   - Model checkpoints in `checkpoint_dir`.
   - Best model saved based on validation Mean IoU.

---

## Requirements

- **Libraries**: `torch`, `torch.utils.tensorboard`, `tqdm`, `json`
- **Framework**: Python 3, PyTorch

---

## Example Usage

```python
# Example setup
from torch.utils.data import DataLoader
from your_model import YourModel  # replace with actual model
from your_dataset import YourDataset  # replace with actual dataset

model = YourModel()
train_loader = DataLoader(YourDataset(split="train"), batch_size=32, shuffle=True)
val_loader = DataLoader(YourDataset(split="val"), batch_size=32)

# Run training
train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    num_classes=5,
    weights=(0.7, 0.3),
    checkpoint_dir="checkpoints",
    log_dir="logs"
)
