import os
import torch
from torch_geometric.data import DataLoader
from src.model import ASLGCN
from sklearn.metrics import accuracy_score, classification_report


TEST_DIR = "data/graphs_test"
files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.endswith(".pt")]

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx])

test_dataset = TestDataset(files)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLGCN(in_channels=2, hidden_channels=64, out_channels=29).to(device)
model.load_state_dict(torch.load("models/model.pth"))
model.eval()

# Predict
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds = out.argmax(dim=1).cpu()
        labels = batch.y.cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")

# Detailed report
print(classification_report(all_labels, all_preds, digits=3))
