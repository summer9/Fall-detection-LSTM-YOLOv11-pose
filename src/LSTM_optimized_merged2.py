# Binary Fall Detection (Falling vs Non-Falling) with Sliding Windows
# Dataset = 30fps pose keypoints
# Softmax probabilities for 2 classes
# → Now using 3-layer LSTM
# → Best model saved based on Falling F1-score (better balance precision/recall, reduces false alarms)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import numpy as np
import random
import datetime
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import recall_score, f1_score, precision_score

# Set seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------- DATASET -------------------
class PoseDataset(Dataset):
    def __init__(self, csv_file, window_size=90, stride=30):
        df = pd.read_csv(csv_file)

        pose_cols = [c for c in df.columns if c.endswith(('_x', '_y', '_c'))]
        assert len(pose_cols) == 51, f"Expected 51 pose columns, found {len(pose_cols)}"

        self.features = df[pose_cols].values.astype(np.float32)
        
        # Binary mappingf
        class_map = {'non-fall': 0, 'fall': 1}
        labels_series = df['final_class'].map(class_map)
        if labels_series.isnull().any():
            raise ValueError("Unknown class found in CSV")
        self.labels = labels_series.values.astype(np.int64)

        self.sequences = []
        self.seq_labels = []

        fall_lengths = []
        for video_path in df['video_path'].unique():
            mask = df['video_path'] == video_path
            video_labels = self.labels[mask]
            if np.any(video_labels == 1):
                fall_indices = np.where(video_labels == 1)[0]
                fall_lengths.append(len(fall_indices))
        print("Fall segment lengths in this CSV (frames):", fall_lengths)

        for video_path in df['video_path'].unique():
            mask = df['video_path'] == video_path
            video_features = self.features[mask]
            video_labels   = self.labels[mask]

            if len(video_features) < window_size:
                continue

            for start in range(0, len(video_features) - window_size + 1, stride):
                end = start + window_size
                
                window_features = video_features[start:end]
                window_labels   = video_labels[start:end]

                fall_ratio = np.mean(window_labels == 1)
                label = 1 if fall_ratio >= 0.3 else 0
                

                self.sequences.append(torch.tensor(window_features, dtype=torch.float32))
                self.seq_labels.append(label)

        print(f"{csv_file} → {len(self.sequences)} sliding windows")
        print("Class distribution:", np.bincount(self.seq_labels))
        print(f"Window size: {window_size}, Stride: {stride}, "
              f"Overlap: {100 * (window_size - stride) / window_size:.0f}%")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.seq_labels[idx]


# ------------------- MODEL -------------------
class FallingBinaryLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=51,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])


# ------------------- TRAINING -------------------
def train_model(train_csv, test_csv, 
                num_epochs=40, 
                batch_size=16, 
                lr=0.0005,
                window_size=90, 
                stride=30):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    run_name = f"2class_fall_3layers_win{window_size}_str{stride}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"runs/{run_name}")

    train_dataset = PoseDataset(train_csv, window_size=window_size, stride=stride)
    test_dataset  = PoseDataset(test_csv,  window_size=window_size, stride=stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    model = FallingBinaryLSTM().to(device)

    weights = torch.tensor([1.0,2.1], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print("Loss Weights (non-fall/ fall):", weights.tolist())

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_f1 = 0.0                   # ← Now tracking best Falling F1
    best_preds  = None
    best_labels = None

    patience = 10
    no_improve = 0
    best_f1_value = 0.0

    class_names = ['non-fall', 'fall']

    for epoch in range(1, num_epochs + 1):

        model.train()
        train_loss = 0.0

        for seqs, labels in train_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)

        train_loss /= len(train_dataset)

        # ---------- TEST ----------
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for seqs, labels in test_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)

                outputs = model(seqs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * labels.size(0)

                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_dataset)
        
        acc         = accuracy_score(all_labels, all_preds)
        fall_precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        fall_recall    = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        fall_f1        = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

        print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Test {test_loss:.4f} | "
              f"Acc {acc:.4f} | Fall Prec {fall_precision:.4f} | Fall Rec {fall_recall:.4f} | "
              f"Fall F1 {fall_f1:.4f}")

        # Save based on Falling F1
        improved = False
        if fall_f1 > best_f1:
            best_f1 = fall_f1
            improved = True

        if improved:
            best_preds = all_preds.copy()
            best_labels = all_labels.copy()
            torch.save(model.state_dict(), "BEST_MODEL_Fall_Nonfall_LSTM_3layers_f1_30fps_optimized.pth")
            print(f"→ New best model saved at epoch {epoch} (Fall F1: {fall_f1:.4f})")

        # Early stopping based on F1
        if fall_f1 > best_f1_value:
            best_f1_value = fall_f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no F1 improvement)")
                break

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Test", acc, epoch)
        writer.add_scalar("Precision/Fall", fall_precision, epoch)
        writer.add_scalar("Recall/Fall", fall_recall, epoch)
        writer.add_scalar("F1/Fall", fall_f1, epoch)

    # ---------- FINAL REPORT ----------
    print("\nBest Fall F1:", round(best_f1, 4))

    cm = confusion_matrix(best_labels, best_preds, labels=[0, 1])

    print("\nConfusion Matrix")
    print("True → / Pred ↓")
    print("              non-fall   fall")
    print(f"non-fall       {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"fall          {cm[1,0]:6d}   {cm[1,1]:6d}")

    print("\nClassification Report:")
    print(classification_report(best_labels, best_preds, target_names=class_names, digits=4))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names,
                cmap='Blues')
    plt.title("Confusion Matrix – Binary Fall Detection (best by Falling F1)")
    plt.show()

    writer.close()
    return model


# ------------------- MAIN -------------------
if __name__ == "__main__":

    model = train_model(
        train_csv="GMDCSA24_30fps_train_stratify_merged2.csv",
        #GMDCSA24_yolo11pose_30fps_gpu_with_final_class_train.csv
        #GMDCSA24_30fps_train.csv
        test_csv="GMDCSA24_30fps_test_stratify_merged2.csv",
        #GMDCSA24_yolo11pose_30fps_gpu_with_final_class_test.csv
        #GMDCSA24_30fps_test.csv
        num_epochs=40,
        batch_size=16,
        lr=0.0001,
        window_size=90,
        stride=30
    )