# Define the model.
import torch


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = torch.nn.Conv1d(input_dim, 64)
        self.fc2 = torch.nn.Conv1d(64, 32)
        self.fc3 = torch.nn.Conv1d(32, 1)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x
