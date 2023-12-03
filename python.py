import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "/kaggle/input/projectdata"
save_path = "/kaggle/working"


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.filepaths = glob.glob(os.path.join(data_dir, '**/*.jpg'), recursive=True)
        self.classes = ['real', 'fake']

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]

        # Extract the class name from the filepath
        class_name = self.classes[1] if "Fake" in os.path.dirname(os.path.relpath(filepath, data_dir)) else \
        self.classes[0]

        # Map the class name to 'real' or 'fake'
        class_label = self.classes.index(class_name)

        # Load and transform the image
        image = Image.open(filepath)

        if self.transform is not None:
            image = self.transform(image)

        return image, class_label

data = {}
data["train"] = CustomDataset(os.path.join(data_dir, "train"), transform = data_transforms["train"])
data["val"] = CustomDataset(os.path.join(data_dir, "val"), transform = data_transforms["val"])

mask = list(range(1, len(data["train"]), 10))
data["train"] = torch.utils.data.Subset(data["train"], mask)

mask = list(range(1, len(data["val"]), 10))
data["val"] = torch.utils.data.Subset(data["val"], mask)

dataloaders = {x: torch.utils.data.DataLoader(data[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(data[x]) for x in ["train", "val"]}

class_names = ['real', 'fake']

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model


model_ft = models.resnet50()
model_ft.load_state_dict(torch.load("resnet50.pth"))

num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Sequential(
    nn.Linear(num_ftrs, num_ftrs//2),
    nn.ReLU(),
    nn.Linear(num_ftrs//2, len(class_names)),
)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=150)

torch.save(model_ft.state_dict(), os.path.join(save_path, "resnet50.pt"))


model_ft = models.efficientnet_b6()
model_ft.load_state_dict(torch.load("efficientnet.pth"))

num_ftrs = model_ft.classifier[1].in_features

model_ft.classifier = nn.Sequential(
    nn.Linear(num_ftrs, num_ftrs//2),
    nn.ReLU(),
    nn.Linear(num_ftrs//2, len(class_names)),
)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=150)

torch.save(model_ft.state_dict(), os.path.join(save_path, "efficientnet_b4.pt"))
