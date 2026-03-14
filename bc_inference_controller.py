import time
import numpy as np
import torch

EPS = 1e-6


class BCInferenceController:
    def __init__(
        self,
        model_path,
        scaler_mean_path,
        scaler_std_path,
        wheelbase,
        dt=0.01,
        device="cpu",
        steer_limit=0.785,   # ~30 deg jugar con esto
        throttle_limit=3.0,
    ):
        self.device = torch.device(device)

        # =========================
        # Load normalization stats
        # =========================
        self.obs_mean = np.load(scaler_mean_path)
        self.obs_std = np.load(scaler_std_path)

        # =========================
        # Load model
        # =========================
        model = torch.load(model_path, map_location=self.device)

        if isinstance(model, dict):
            # state_dict case
            from train_bc import BCPolicy   # AJUSTA si el nombre cambia
            self.net = BCPolicy(input_dim=len(self.obs_mean))
            self.net.load_state_dict(model)
        else:
            self.net = model

        self.net.to(self.device)
        self.net.eval()

        # =========================
        # Geometry & timing
        # =========================
        self.wheelbase = wheelbase
        self.dt = dt

        # =========================
        # LiDAR layout (IDENTICAL)
        # =========================
        self.n_left = 10
        self.n_front = 30
        self.n_right = 10
        self.n_sectors = 50

        self.left_sector_size = 36
        self.front_sector_size = 12
        self.right_sector_size = 36

        self.prev_mean_ranges = np.zeros(self.n_sectors)

        # =========================
        # Temporal buffers
        # =========================
        self.prev_steer = 0.0
        self.prev2_steer = 0.0
        self.prev_speed = 0.0
        self.prev2_speed = 0.0
        self.last_time = time.time()

        # =========================
        # Limits
        # =========================
        self.steer_limit = steer_limit
        self.throttle_limit = throttle_limit

    # --------------------------------------------------
    def _sectorize(self, ranges):
        sectors = []
        idx = 0

        for _ in range(self.n_left):
            sectors.append(ranges[idx:idx + self.left_sector_size])
            idx += self.left_sector_size

        for _ in range(self.n_front):
            sectors.append(ranges[idx:idx + self.front_sector_size])
            idx += self.front_sector_size

        for _ in range(self.n_right):
            sectors.append(ranges[idx:idx + self.right_sector_size])
            idx += self.right_sector_size

        return sectors

    # --------------------------------------------------
    def compute_features(self, obs):
        ranges = np.array(obs["scans"][0])
        ranges = np.clip(ranges, 0.0, 30.0)

        sectors = self._sectorize(ranges)

        feats = []
        mean_ranges = []

        # =========================
        # LiDAR sector stats
        # =========================
        for i, s in enumerate(sectors):
            valid = np.isfinite(s)
            sv = s[valid]
            if len(sv) == 0:
                sv = np.array([30.0])

            mn = np.min(sv)
            mu = np.mean(sv)
            sd = np.std(sv)
            p10, p25, p50, p75, p90 = np.percentile(sv, [10, 25, 50, 75, 90])
            vr = len(sv) / len(s)
            inv = np.mean(1.0 / (sv + EPS))

            self.prev_mean_ranges[i] = mu
            mean_ranges.append(mu)

            feats += [mn, mu, sd, p10, p25, p50, p75, p90, vr, inv]

        # =========================
        # Global LiDAR
        # =========================
        left = mean_ranges[0:self.n_left]
        front = mean_ranges[self.n_left:self.n_left + self.n_front]
        right = mean_ranges[self.n_left + self.n_front:]

        left_fs = np.mean(left)
        right_fs = np.mean(right)
        front_fs = np.percentile(front, 90)

        track_width = np.median(left) + np.median(right)
        center_offset = (np.median(right) - np.median(left)) / 2.0

        feats += [
            left_fs,
            right_fs,
            front_fs,
            left_fs - right_fs,
            track_width,
            center_offset,
        ]

        # =========================
        # Vehicle dynamics
        # =========================
        vx = obs["linear_vels_x"][0]
        vy = obs["linear_vels_y"][0]
        speed = np.sqrt(vx**2 + vy**2)
        yaw_rate = obs["ang_vels_z"][0]
        yaw_rate_os = yaw_rate / max(speed, EPS)

        feats += [speed, yaw_rate, yaw_rate_os]
        # debug
        # print("N features:", len(feats))

        return np.array(feats, dtype=np.float32), speed

    # --------------------------------------------------
    def act(self, obs):
        x, speed = self.compute_features(obs)

        # =========================
        # Manual StandardScaler
        # =========================
        x = (x - self.obs_mean) / (self.obs_std + EPS)
        x = torch.from_numpy(x).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.net(x).cpu().numpy()[0]

        steer = np.clip(out[0], -self.steer_limit, self.steer_limit)
        throttle = np.clip(out[1], 0.0, self.throttle_limit)
        # print(f"speed={speed:.3f} steer={steer:.3f}")

        # update buffers
        self.prev2_steer = self.prev_steer
        self.prev_steer = steer
        self.prev2_speed = self.prev_speed
        self.prev_speed = speed

        return np.array([[steer, throttle]], dtype=np.float32)

    
