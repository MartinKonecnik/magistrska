# Martin Konečnik, https://git.siwim.si/machine-learning/fix-qa-binary-classification
# Finds the best seed based on F3 metric.
import pickle
import tomllib
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from classifier import BinaryClassifier

# Initialize CUDA.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Read the configuration file.
with open('conf.toml', 'rb') as f:
    conf = tomllib.load(f)

LOAD_MODEL = conf['model']
SEEDS = conf['seeds']

# Load test data
unaltered_pkl_path = Path('test_data') / 'unaltered.pkl'
corrected_pkl_path = Path('test_data') / 'corrected.pkl'

with open(unaltered_pkl_path, 'rb') as f:
    test_unaltered = pickle.load(f)

with open(corrected_pkl_path, 'rb') as f:
    test_corrected = pickle.load(f)

# Store results
results = []

for seed in SEEDS:
    model_name = f"{LOAD_MODEL}-{seed}"
    model_path = Path('models', model_name)

    # Load max_length from model
    dim = int(open(Path(model_path, 'dimensions')).read())


    # Pad test signals to match model's max_length
    def pad_signal(sig, target_length):
        if len(sig) >= target_length:
            return sig[:target_length]
        else:
            return np.pad(sig, (0, target_length - len(sig)), 'constant')


    # Povprečenje čez vse kanale (enako kot pri učnih podatkih v celici [6])
    test_corrected_avg = np.mean(test_corrected, axis=0)  # [n_test_corrected, max_length]
    test_unaltered_avg = np.mean(test_unaltered, axis=0)  # [n_test_unaltered, max_length]

    # Priprava podatkov za model
    X_test = np.vstack((test_corrected_avg, test_unaltered_avg))
    y_test = np.array([1] * len(test_corrected_avg) + [0] * len(test_unaltered_avg))

    # Create data loader
    X_tensor = torch.from_numpy(X_test).unsqueeze(1).float().to(device)
    y_tensor = torch.from_numpy(y_test).float().to(device)
    test_dataset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load model
    model = BinaryClassifier(input_dim=dim).to(device)
    model.load_state_dict(torch.load(Path(model_path, 'fix-qa-binary-classification.pth'), map_location=device))
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            all_preds.extend(predicted.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f3 = fbeta_score(all_labels, all_preds, beta=3, zero_division=0)

    results.append({
        'seed': seed,
        'precision': precision,
        'recall': recall,
        'f3': f3
    })

# Sort by F3
results.sort(key=lambda x: x['f3'], reverse=True)

# Print table
print("\n" + "-" * 60)
print(f"{'Rank':<6} {'Seed':<10} {'Precision':<12} {'Recall':<12} {'F3':<12}")
print("-" * 60)
for rank, r in enumerate(results, 1):
    print(f"{rank:<6} {r['seed']:<10} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f3']:<12.4f}")
print("-" * 60)

# Print best
print(f"\nBest model: Seed {results[0]['seed']} with F3 = {results[0]['f3']:.4f}")
