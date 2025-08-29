# Define the model.
import torch


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        # Input shape should be: [batch_size, 1, input_dim]
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        print(f'Input shape: {x.shape}')  # Debugging

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return self.sigmoid(x)
