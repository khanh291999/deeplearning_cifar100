import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directory to save models
os.makedirs('models', exist_ok=True)

# Enhanced data augmentation for better accuracy
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    transforms.RandomErasing(p=0.1)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)  # Smaller batch size for better convergence
testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)  # Smaller test batch size

# load pretrained model and modify last layer
model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')  # Updated syntax
model.fc = torch.nn.Linear(model.fc.in_features, 100)
model = model.to(device)

# Enhanced optimizer and scheduler for high accuracy
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
# Cosine annealing with warm restarts for better convergence
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

# Train the model with improved settings for high accuracy
print("Starting training...")
num_epochs = 100  # Increased epochs for better results (aim for 90% accuracy)
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}, '
                  f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    # Update learning rate
    scheduler.step()
    
    epoch_acc = 100.*correct/total
    epoch_loss = running_loss/len(trainloader)
    print(f'Epoch {epoch+1}/{num_epochs} completed. Loss: {epoch_loss:.4f}, '
          f'Training Accuracy: {epoch_acc:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Validation every 10 epochs to monitor progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100.*val_correct/val_total
        print(f'Validation Accuracy after epoch {epoch+1}: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        # Early stopping if we achieve 90%+ accuracy
        if val_acc >= 90.0:
            print(f"🎉 Target accuracy of 90% achieved! Validation Accuracy: {val_acc:.2f}%")
            break

print("Training completed!")

# evaluation
print("Starting evaluation...")
model.eval()
all_preds, all_targets = [], []
all_outputs = []

with torch.no_grad():
    for images, targets in testloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_targets.append(targets)
        all_outputs.append(outputs.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()
all_outputs = torch.cat(all_outputs).numpy()

# confusion matrix
cm = confusion_matrix(all_targets, all_preds)

# accuracy, precision, recall, F1
precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
accuracy = (all_preds == all_targets).mean()

# top-5 metrics
top5_acc = top_k_accuracy_score(all_targets, all_outputs, k=5)
# precision@5 calculation (simplified)
precision_at5 = top5_acc / 5
f1_top5 = 2 * (precision_at5 * top5_acc) / (precision_at5 + top5_acc) if (precision_at5 + top5_acc) > 0 else 0

print(f"\nResults:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")
print(f"Precision@5: {precision_at5:.4f}")
print(f"F1@5: {f1_top5:.4f}")

# Plot confusion matrix (sample of first 20 classes to make it readable)
plt.figure(figsize=(12, 10))
cm_sample = cm[:20, :20]  # Show only first 20x20 classes for readability
sns.heatmap(cm_sample, annot=True, fmt='d', cmap='Blues', square=True)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('CIFAR-100 Confusion Matrix (First 20 Classes)')
plt.show()

print("Evaluation completed!")

print(f"\\n🎯 Best validation accuracy achieved: {best_acc:.2f}%")
print("\\n📝 Key improvements made for 90% accuracy:")
print("✅ Enhanced data augmentation (rotation, color jitter, random erasing)")
print("✅ Label smoothing for better generalization") 
print("✅ Cosine annealing learning rate scheduler")
print("✅ Smaller batch sizes for better convergence")
print("✅ Early stopping when target accuracy reached")
print("✅ Model checkpointing (best model saved)")
print("\\n💡 Additional tips for even higher accuracy:")
print("- Use GPU for faster training (current: CPU)")
print("- Try different architectures (EfficientNet, Vision Transformer)")
print("- Use mixed precision training")
print("- Ensemble multiple models")
print("- Fine-tune for more epochs with lower learning rate")
