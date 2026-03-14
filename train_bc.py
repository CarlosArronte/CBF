import glob
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class BCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BCPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


def get_feature_and_target_cols():
    # ==============================
    # TARGETS
    # ==============================
    target_cols = ["steering_cmd", "throttle_cmd"]

    # ==============================
    # BC-SAFE FEATURES (EXPLICIT)
    # ==============================
    feature_cols = []

    # --- LiDAR sectors ---
    n_sectors = 50
    lidar_feats = [
        "lidar_min",
        "lidar_mean",
        "lidar_std",
        "lidar_p10",
        "lidar_p25",
        "lidar_p50",
        "lidar_p75",
        "lidar_p90",
        "lidar_valid_ratio",
        "lidar_mean_inv",
    ]

    for i in range(n_sectors):
        for f in lidar_feats:
            feature_cols.append(f"{f}_s{i}")

    # --- Global LiDAR geometry ---
    feature_cols += [
        "left_free_space",
        "right_free_space",
        "front_free_space",
        "free_space_lr",
        "track_width_est",
        "center_offset_lidar",
    ]

    # --- Instantaneous vehicle state ---
    feature_cols += [
        "speed",
        "yaw_rate",
        "yaw_rate_over_speed",
    ]

    return feature_cols, target_cols


def load_dataset_file(file_path, feature_cols, target_cols):
    use_cols = feature_cols + target_cols
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path, usecols=use_cols)

    # ==============================
    # CLEAN DATA
    # ==============================
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    after = len(df)

    print(
        f"Dropped {before - after} invalid rows "
        f"({100*(before-after)/before:.2f}%)"
    )

    # ==============================
    # BUILD MATRICES (ORDER MATTERS)
    # ==============================
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    return X, y




def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==============================
    # LOAD DATA
    # ==============================
    ds_path = "DS"
    files = sorted(glob.glob(f"{ds_path}/*_DS.csv"))
    assert len(files) > 0, "No DS files found"

    feature_cols, target_cols = get_feature_and_target_cols()

    # ==============================
    # GLOBAL SCALER (STREAMING)
    # ==============================
    scaler = StandardScaler()

    for f in files:
        X, y = load_dataset_file(f, feature_cols, target_cols)

        # debug
        assert np.isfinite(X).all(), "X still contains NaN or Inf"
        assert np.isfinite(y).all(), "y still contains NaN or Inf"

        scaler.partial_fit(X)

    joblib.dump(scaler, "bc_scaler.pkl")
    np.save("bc_scaler_mean.npy", scaler.mean_)
    np.save("bc_scaler_std.npy", scaler.scale_)

    # ==============================
    # MODEL
    # ==============================
    model = BCPolicy(input_dim=len(feature_cols)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Steering más importante
    loss_weights = torch.tensor([2.0, 1.0], device=device)
    criterion = nn.MSELoss(reduction="none")

    # ==============================
    # TRAIN LOOP
    # ==============================
    epochs_per_ds = 20

    for f in files:
        X, y = load_dataset_file(f, feature_cols, target_cols)
        X = scaler.transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42, shuffle=True
        )

        train_ds = BCDataset(X_train, y_train)
        val_ds = BCDataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

        for epoch in range(epochs_per_ds):
            model.train()
            train_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb)
                loss = criterion(pred, yb)
                loss = (loss * loss_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # --------------------------
            # VALIDATION
            # --------------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)

                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss = (loss * loss_weights).mean()
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(
                f"File {Path(f).name} | "
                f"Epoch {epoch:03d} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )

    torch.save(model.state_dict(), "bc_policy.pt")
    print("Training finished.")


if __name__ == "__main__":
    train()
