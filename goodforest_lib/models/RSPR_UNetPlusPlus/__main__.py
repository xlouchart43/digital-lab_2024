import os
import sys
from datetime import datetime
import numpy as np
import json
from model_no_normalization import RSPRUNetPlusPlus
from training_masked import train
from data_loaders import get_loaders


if __name__ == "__main__":

    file_path = "train_4-sick-classes_IB_set.h5"  # Path to the training dataset
    val_file_path = (
        "validation_4-sick-classes_IB_set.h5"  # Path to the validation dataset
    )
    batch_size = 12
    train_split = 0.8
    num_epochs = 200
    num_classes = 5
    weights = [
        0.4,
        0.4,
    ]  # dice_weight, cross_entropy_weight : weights for the loss function. If the sum of the weights is 1, there will be not false positive penalty

    # Change everything below this line to match your dataset
    data_description = {
        "file_path": file_path,
        "num_classes": num_classes,
        "input_channels": 24,
        "description": "concat outputs, no normalization, 4 classes as below, fp penalty, masked background, ",
        "classes": [
            "Class 255 is background",
            "Class 0 is healthy ",
            "Class 1 is T ",
            "Class 2 is L ",
            "Class 3 is M",
            "Class 4 is S + V",
        ],
    }
    model_description = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "weights": weights,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"training_run_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    checkpoint_dir = os.path.join(run_folder, "checkpoints")
    log_dir = os.path.join(run_folder, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_loader, val_loader, norm_stats = get_loaders(
        file_path, val_file_path, batch_size, train_split, normalize=False
    )

    for batch in train_loader:
        data, label = batch
        print(data.shape)
        print(label.shape)
        print(np.unique(label))
        break
    model = RSPRUNetPlusPlus(
        num_classes=num_classes, input_channels=data_description["input_channels"]
    )

    data_description["num_cubes"] = len(train_loader.dataset) + len(val_loader.dataset)
    data_description["train_split"] = train_split

    with open(os.path.join(run_folder, "data_description.json"), "w") as f:
        json.dump(data_description, f, indent=4)
    with open(os.path.join(run_folder, "model_description.json"), "w") as f:
        json.dump(model_description, f, indent=4)

    print("Starting training")

    train(
        model,
        train_loader,
        val_loader,
        num_epochs,
        num_classes,
        weights,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        resume_checkpoint=None,  # If you want to resume training from a checkpoint, specify the path here
        data_description=data_description,
        model_description=model_description,
    )
