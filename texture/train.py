import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

from tqdm import tqdm

# ==========================================================
# SETTINGS
# ==========================================================

# ==========================================================
# SETTINGS
# ==========================================================

TRAIN_DIR = r"C:\texture\dataset_split\train"
VAL_DIR = r"C:\texture\dataset_split\val"

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 40

LEARNING_RATE = 1e-4

DEVICE = torch.device("cpu")

SAVE_MODEL = "best_model.pth"

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

train_transform = transforms.Compose([

    transforms.Resize((256,256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.7,1.0)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.15,
        hue=0.05
    ),

    transforms.RandomPerspective(
        distortion_scale=0.20,
        p=0.30
    ),

    transforms.GaussianBlur(3),

    transforms.RandomGrayscale(p=0.05),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

val_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ==========================================================
# DATASET
# ==========================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

class_names = train_dataset.classes

print("\n===============================")
print("Classes")
print("===============================")

for idx, name in enumerate(class_names):
    print(f"{idx} : {name}")

print("\nTraining Images :", len(train_dataset))
print("Validation Images :", len(val_dataset))

# ==========================================================
# MODEL
# ==========================================================

print("\nLoading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

# Freeze feature extractor initially
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(class_names)
)

model = model.to(DEVICE)

print(model)

# ==========================================================
# LOSS FUNCTION
# ==========================================================

criterion = nn.CrossEntropyLoss()

# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# ==========================================================
# LEARNING RATE SCHEDULER
# ==========================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# ==========================================================
# TRAINING VARIABLES
# ==========================================================

best_accuracy = 0.0

best_weights = copy.deepcopy(model.state_dict())

train_accuracy_history = []

validation_accuracy_history = []

train_loss_history = []

validation_loss_history = []

# Early stopping
EARLY_STOPPING_PATIENCE = 7

early_stop_counter = 0

print("\n=====================================")
print("Training Configuration")
print("=====================================")

print(f"Device              : {DEVICE}")
print(f"Epochs              : {EPOCHS}")
print(f"Batch Size          : {BATCH_SIZE}")
print(f"Learning Rate       : {LEARNING_RATE}")
print(f"Training Images     : {len(train_dataset)}")
print(f"Validation Images   : {len(val_dataset)}")
print(f"Number of Classes   : {len(class_names)}")

print("\nModel Ready.\n")

# ==========================================================
# TRAINING LOOP
# ==========================================================

for epoch in range(EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("=" * 60)

    ###############################################
    # TRAIN
    ###############################################

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(train_loader)

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predictions = torch.max(outputs, 1)

        train_correct += (predictions == labels).sum().item()

        train_total += labels.size(0)

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = train_loss / len(train_loader)

    train_accuracy = train_correct / train_total

    ###############################################
    # VALIDATION
    ###############################################

    model.eval()

    validation_loss = 0.0

    validation_correct = 0

    validation_total = 0

    y_true = []

    y_pred = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            _, predictions = torch.max(outputs, 1)

            validation_correct += (predictions == labels).sum().item()

            validation_total += labels.size(0)

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(predictions.cpu().numpy())

    validation_loss = validation_loss / len(val_loader)

    validation_accuracy = validation_correct / validation_total

    ###############################################
    # SAVE HISTORY
    ###############################################

    train_loss_history.append(train_loss)

    validation_loss_history.append(validation_loss)

    train_accuracy_history.append(train_accuracy)

    validation_accuracy_history.append(validation_accuracy)

    ###############################################
    # PRINT RESULTS
    ###############################################

    print(f"\nTrain Loss        : {train_loss:.4f}")

    print(f"Validation Loss   : {validation_loss:.4f}")

    print(f"Train Accuracy    : {train_accuracy*100:.2f}%")

    print(f"Validation Accuracy : {validation_accuracy*100:.2f}%")

    ###############################################
    # LEARNING RATE SCHEDULER
    ###############################################

    scheduler.step(validation_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    print(f"Learning Rate     : {current_lr:.8f}")

    ###############################################
    # SAVE BEST MODEL
    ###############################################

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        best_weights = copy.deepcopy(model.state_dict())

        torch.save(best_weights, SAVE_MODEL)

        early_stop_counter = 0

        print("\n✅ Best model saved!")

    else:

        early_stop_counter += 1

        print(
            f"Early Stopping Counter : "
            f"{early_stop_counter}/{EARLY_STOPPING_PATIENCE}"
        )

    ###############################################
    # EARLY STOPPING
    ###############################################

    if early_stop_counter >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered.")

        break

# ==========================================================
# LOAD BEST MODEL
# ==========================================================

model.load_state_dict(best_weights)

print("\nTraining Finished.")

print(f"\nBest Validation Accuracy : {best_accuracy*100:.2f}%")

# ==========================================================
# FINAL EVALUATION
# ==========================================================

print("\n")
print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix\n")

print(cm)

# ==========================================================
# SAVE CONFUSION MATRIX
# ==========================================================

plt.figure(figsize=(8,8))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            str(cm[i][j]),
            ha="center",
            va="center"
        )

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

# ==========================================================
# ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    train_accuracy_history,
    label="Train Accuracy",
    linewidth=2
)

plt.plot(
    validation_accuracy_history,
    label="Validation Accuracy",
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title("Training Accuracy")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("accuracy.png")

# ==========================================================
# LOSS GRAPH
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    train_loss_history,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    validation_loss_history,
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig("loss.png")

plt.show()

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nBest Validation Accuracy : {best_accuracy*100:.2f}%")

print("\nFiles Saved")

print("-------------------------")

print("best_model.pth")

print("accuracy.png")

print("loss.png")

print("confusion_matrix.png")

print("classification_report.txt")