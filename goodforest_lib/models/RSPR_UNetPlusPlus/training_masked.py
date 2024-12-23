import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from typing import Tuple


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 0.1
) -> torch.Tensor:
    """
    Calculate the Dice loss between predicted and target masks.

    Args:
        pred (torch.Tensor): Predicted mask of shape (N, C, H, W).
        target (torch.Tensor): Target mask of shape (N, H, W).
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Dice loss."""
    mask = target != 255
    mask_expand = mask.unsqueeze(1).expand_as(pred)

    pred_ = pred * mask_expand
    target_one_hot = (
        F.one_hot(torch.clamp(target, max=pred.size(1) - 1), num_classes=pred_.size(1))
        .permute(0, 3, 1, 2)
        .float()
    )

    target_one_hot = target_one_hot * mask_expand

    intersection = (pred_ * target_one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (
        pred_.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3)) + smooth
    )
    return 1 - dice.mean()


def combined_loss(
    pred: torch.Tensor, target: torch.Tensor, weights: Tuple[float, float]
) -> torch.Tensor:
    """
    Calculate the combined loss of Dice loss,Cross-Entropy loss and False Positive penalty.

    Args:
        pred (torch.Tensor): Predicted mask of shape (N, C, H, W).
        target (torch.Tensor): Target mask of shape (N, H, W).
        weights (tuple): Weights for Dice loss and Cross-Entropy loss.

    Returns:
        torch.Tensor: Combined loss."""
    dice_weight, ce_weight = weights
    dice = dice_loss(F.softmax(pred, dim=1), target)
    mask = target != 255
    valid_target = target[mask]

    class_weights = torch.tensor(
        [1.0, 10.0, 5.0, 3.0, 2.0], dtype=torch.float32, device=pred.device
    )  # Change class weights for the cross-entropy loss
    ce = F.cross_entropy(pred, target, weight=class_weights, ignore_index=255)

    if dice_weight + ce_weight < 1:
        softmax_pred = F.softmax(pred, dim=1)
        target_one_hot = (
            F.one_hot(
                torch.clamp(target, max=pred.size(1) - 1), num_classes=pred.size(1)
            )
            .permute(0, 3, 1, 2)
            .float()
        )
        mask_expand = mask.unsqueeze(1).expand_as(pred)

        fp_penalty = torch.sum(softmax_pred * (1 - target_one_hot) * mask_expand) / (
            mask.sum() * pred.size(1)
        )

        total_loss = (
            dice_weight * dice
            + ce_weight * ce
            + (1 - dice_weight - ce_weight) * fp_penalty
        )
    else:
        total_loss = dice_weight * dice + ce_weight * ce
    return total_loss


def calculate_metrics(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict:
    """
    Calculate the IoU, F1, Precision, and Recall for each class.

    Args:
        preds (torch.Tensor): Predicted mask of shape (N, H, W).
        labels (torch.Tensor): Target mask of shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary containing the metrics for each class."""
    metrics = {}
    mask = labels != 255
    for cls in range(num_classes):
        pred_cls = (preds == cls) & mask
        label_cls = (labels == cls) & mask
        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        precision = (intersection + 1e-6) / (pred_cls.sum() + 1e-6)
        recall = (intersection + 1e-6) / (label_cls.sum() + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        metrics[f"IoU_class_{cls}"] = iou.item()
        metrics[f"F1_class_{cls}"] = f1.item()
        metrics[f"Precision_class_{cls}"] = precision.item()
        metrics[f"Recall_class_{cls}"] = recall.item()

    return metrics


def validate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, dict]:
    """
    Validate the model on the validation set.

    Args:
        model (torch.nn.Module): Model to validate.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion: Loss function.
        device (torch.device): Device to run the validation on.
        num_classes (int): Number of classes.

    Returns:
        float: Average loss on the validation set.
        dict: Dictionary containing the metrics for the validation set.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    avg_loss = total_loss / len(val_loader.dataset)

    metrics = calculate_metrics(all_preds, all_labels, num_classes)

    accuracy = (all_preds == all_labels).float().mean().item()
    metrics["Overall_Accuracy"] = accuracy

    mean_iou = sum([metrics[f"IoU_class_{i}"] for i in range(num_classes)]) / (
        num_classes
    )
    metrics["Mean_IoU"] = mean_iou

    return avg_loss, metrics


def log_validation_results(
    writer: SummaryWriter, metrics: dict, loss: float, epoch: int
) -> None:
    """
    Log the validation results to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard SummaryWriter.
        metrics (dict): Dictionary containing the metrics for the validation set.
        loss (float): Average loss on the validation set.
        epoch (int): Current epoch.
    """
    writer.add_scalar("Validation/Loss", loss, epoch)
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f"Validation/{metric_name}", metric_value, epoch)


def validation_epoch(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion,
    device: torch.device,
    num_classes: int,
    epoch: int,
    writer: SummaryWriter,
) -> Tuple[float, dict]:
    """
    Run a validation epoch and log the results.

    Args:
        model (torch.nn.Module): Model to validate.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion: Loss function.
        device (torch.device): Device to run the validation on.
        num_classes (int): Number of classes.
        epoch (int): Current epoch.
        writer (SummaryWriter): TensorBoard SummaryWriter.

    Returns:
        float: Average loss on the validation set.
        dict: Dictionary containing the metrics for the validation set."""
    print(f"\nValidation Epoch: {epoch}")

    val_loss, val_metrics = validate(model, val_loader, criterion, device, num_classes)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Mean IoU: {val_metrics['Mean_IoU']:.4f}")
    print(f"Validation Accuracy: {val_metrics['Overall_Accuracy']:.4f}")

    for cls in range(num_classes):
        print(
            f"Class {cls} - IoU: {val_metrics[f'IoU_class_{cls}']:.4f}, "
            f"F1: {val_metrics[f'F1_class_{cls}']:.4f}"
        )

    log_validation_results(writer, val_metrics, val_loss, epoch)

    return val_loss, val_metrics


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    num_classes: int,
    weights: Tuple[float, float],
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    resume_checkpoint: str = None,
    data_description: str = None,
    model_description: str = None,
):
    """
    Train the model on the training set and validate on the validation set.

    Args:
        model: torch.nn.Module: Model to train.
        train_loader: torch.utils.data.DataLoader: Training data loader.
        val_loader: torch.utils.data.DataLoader: Validation data loader.
        num_epochs: int: Number of epochs to train the model.
        num_classes: int: Number of classes.
        weights: tuple: Weights for Dice loss and Cross-Entropy loss.
        checkpoint_dir: str: Directory to save the model checkpoints.
        log_dir: str: Directory to save the TensorBoard logs.
        resume_checkpoint: str: Path to the checkpoint to resume training.
        data_description: dict: Description of the dataset.
        model_description: dict: Description of the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    print("Model moved to device")
    criterion = lambda pred, target: combined_loss(pred, target, weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2, verbose=True
    )
    print("Optimizer and scheduler initialized")
    writer = SummaryWriter(log_dir)
    if data_description:
        writer.add_text("Data Description", json.dumps(data_description, indent=2))
    if model_description:
        model_description["device"] = str(device)
        writer.add_text("Model Description", json.dumps(model_description, indent=2))

    best_miou = 0
    best_acc = 0
    global_step = 0
    start_epoch = 0

    # Load checkpoint if specified
    if resume_checkpoint:
        if os.path.isfile(resume_checkpoint):
            print(f"Loading checkpoint '{resume_checkpoint}'")
            checkpoint = torch.load(resume_checkpoint)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_miou = checkpoint["best_miou"]
            best_acc = checkpoint["best_acc"]
            global_step = checkpoint["global_step"]
            print(
                f"Loaded checkpoint '{resume_checkpoint}' (epoch {checkpoint['epoch']})"
            )
        else:
            print(f"No checkpoint found at '{resume_checkpoint}'")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_loss = 0
            epoch_acc = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for i, batch in enumerate(pbar):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

                epoch_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                batch_acc = (preds == labels).float().mean().item()
                epoch_acc += batch_acc

                global_step += 1

                writer.add_scalar("Training/Loss", loss.item(), global_step)
                writer.add_scalar("Training/Accuracy", batch_acc, global_step)
                for param_group in optimizer.param_groups:
                    current_lr = param_group["lr"]
                    writer.add_scalar("Learning Rate", current_lr, global_step)
                pbar.set_postfix({"loss": loss.item(), "accuracy": batch_acc})

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_epoch_acc = epoch_acc / len(train_loader)

            val_loss, val_metrics = validation_epoch(
                model, val_loader, criterion, device, num_classes, epoch, writer
            )

            scheduler.step(val_metrics["Mean_IoU"])

            # Save checkpoint
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_miou": best_miou,
                    "best_acc": best_acc,
                    "global_step": global_step,
                },
                checkpoint_path,
            )

            if val_metrics["Mean_IoU"] > best_miou:
                best_miou = val_metrics["Mean_IoU"]
                best_acc = val_metrics["Overall_Accuracy"]
                best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, "
                f"Train Accuracy: {avg_epoch_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Mean IoU: {val_metrics['Mean_IoU']:.4f}, "
                f"Val Accuracy: {val_metrics['Overall_Accuracy']:.4f}"
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current epoch weights...")
        interrupt_checkpoint_path = os.path.join(
            checkpoint_dir, f"interrupt_checkpoint_epoch_{epoch}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_miou": best_miou,
                "best_acc": best_acc,
                "global_step": global_step,
            },
            interrupt_checkpoint_path,
        )
        print(f"Interrupt checkpoint saved at: {interrupt_checkpoint_path}")
    finally:
        writer.close()
