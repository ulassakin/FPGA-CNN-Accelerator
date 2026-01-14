import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


device = torch.device("cpu")
print("Using device:", device)


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = FashionCNN().to(device)
model.load_state_dict(torch.load("fashioncnn_weights.pth", map_location=device))
model.eval()

print("Model loaded successfully.")


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    transform=test_transform,
    download=True
)


classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

image, label = test_dataset[0]

with torch.no_grad():
    image_input = image.unsqueeze(0).to(device)  # (1, 1, 28, 28)
    output = model(image_input)
    prediction = output.argmax(dim=1).item()

print("Predicted class:", classes[prediction])
print("True class:", classes[label])


