import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, 2),
        )

        self.fc2 = nn.Linear(5408, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.out(x)
        return x

    def get_activations(self, x):
        return self.conv(x)
