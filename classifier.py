# Define the model.
import torch


class BinaryClassifier(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Larger initial kernels to capture broader patterns
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv1d(1, 128, kernel_size=51, padding=25),  # Larger kernel
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv1d(128, 64, kernel_size=25, padding=12),  # Medium kernel
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Conv1d(64, 32, kernel_size=11, padding=5),  # Smaller kernel
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
        )

        # Add attention mechanism to focus on non-zero regions
        self.attention = torch.nn.Sequential(
            torch.nn.Conv1d(32, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )

        self.adaptive_pool = torch.nn.AdaptiveAvgPool1d(1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(16, 1)
        )

    def forward(self, x):
        # Create mask for non-zero values
        non_zero_mask = (x != 0).float()

        # Forward through conv layers
        features = self.conv_layers(x)

        # Apply attention to focus on important regions
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Apply non-zero mask to attended features; his helps the model ignore padded zeros
        if non_zero_mask.shape[2] != attended_features.shape[2]:
            non_zero_mask = torch.nn.functional.interpolate(
                non_zero_mask, size=attended_features.shape[2]
            )
        attended_features = attended_features * non_zero_mask

        # Global pooling and classification
        x = self.adaptive_pool(attended_features)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x.squeeze()
