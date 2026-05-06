import torch


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.relu = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv1d(1, 128, kernel_size=51, padding=25)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout1 = torch.nn.Dropout(0.3)

        self.conv2 = torch.nn.Conv1d(128, 64, kernel_size=25, padding=12)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout2 = torch.nn.Dropout(0.3)

        self.conv3 = torch.nn.Conv1d(64, 64, kernel_size=15, padding=7)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.dropout3 = torch.nn.Dropout(0.25)

        self.conv4 = torch.nn.Conv1d(64, 32, kernel_size=11, padding=5)
        self.bn4 = torch.nn.BatchNorm1d(32)
        self.dropout4 = torch.nn.Dropout(0.2)

        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(32, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )

        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            self.relu,
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        non_zero_mask = (x != 0).float()

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        residual = out
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.relu(out)
        out = self.dropout4(out)

        attention_weights = self.attention(out)
        attended_features = out * attention_weights

        # Mask
        if non_zero_mask.shape[2] != attended_features.shape[2]:
            non_zero_mask = torch.nn.functional.interpolate(
                non_zero_mask, size=attended_features.shape[2], mode='nearest'
            )
        attended_features = attended_features * non_zero_mask

        # Pooling in klasifikacija
        out = self.adaptive_pool(attended_features)
        out = out.squeeze(-1)
        out = self.classifier(out)
        return out.squeeze()
