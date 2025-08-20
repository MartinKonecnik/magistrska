# Define the model.
import torch


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(32, 1, kernel_size=3, padding=1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)  # To get single output

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.adaptive_pool(x)  # Reduce to (batch_size, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # Remove extra dimensions
        x = self.sigmoid(x)
        return x
