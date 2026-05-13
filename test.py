import os
import sys
import random
import time
import datetime
import logging
import struct
import requests
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

CURRENT_DIR = Path(__file__).resolve().parent


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode {mode} not supported")
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - (self.last if self.subtract_last else self.mean)
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + (self.last if self.subtract_last else self.mean)
        return x


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1, 1)
        x = torch.cat([front, x], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        res = x - trend
        return res, trend


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction=4):
        super(SEBlock, self).__init__()
        inner_dim = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, inner_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(inner_dim, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = x.mean(dim=-1, keepdim=True)
        scale = self.fc(scale)
        return x * scale


class FeatureExtractionBlock(nn.Module):
    def __init__(self, d_model: int, patch_len: int, stride: int):
        super(FeatureExtractionBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=d_model * 2,
            kernel_size=patch_len,
            stride=stride,
        )
        self.se = SEBlock(d_model)

    def forward(self, x):
        x = self.conv(x)
        val, gate = x.chunk(2, dim=1)
        x = val * F.gelu(gate)
        return self.se(x)


class FlattenHead(nn.Module):
    def __init__(
        self,
        individual: bool,
        n_vars: int,
        nf: int,
        target_window: int,
        head_dropout=0.0,
    ):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList(
                [nn.Linear(nf, target_window) for _ in range(n_vars)]
            )
            self.dropouts = nn.ModuleList(
                [nn.Dropout(head_dropout) for _ in range(n_vars)]
            )
            self.flattens = nn.ModuleList(
                [nn.Flatten(start_dim=-2) for _ in range(n_vars)]
            )
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(head_dropout)
            self.linear = nn.Linear(nf, target_window)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.dropouts[i](z)
                z = self.linears[i](z)
                x_out.append(z)
            return torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.dropout(x)
            return self.linear(x)


class PatchLinear(nn.Module):
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        d_model=128,
        head_dropout=0.0,
        individual=False,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.c_in = c_in
        self.individual = individual

        pad_len = stride - ((context_window - patch_len) % stride)
        if pad_len == stride:
            pad_len = 0
        self.padding_patch_layer = nn.ReplicationPad1d((0, pad_len))
        self.num_patches = int((context_window + pad_len - patch_len) / stride) + 1
        self.patch_conv = FeatureExtractionBlock(d_model, patch_len, stride)
        self.head_nf = d_model * self.num_patches
        self.head = FlattenHead(
            self.individual,
            self.c_in,
            self.head_nf,
            target_window,
            head_dropout=head_dropout,
        )

    def forward(self, z):
        B, C, L = z.shape

        z = self.padding_patch_layer(z)
        z = z.reshape(B * C, 1, -1)
        z = self.patch_conv(z)
        z = z.reshape(B, C, self.d_model, -1)
        return self.head(z)


class Model(nn.Module):
    def __init__(self, configs, verbose: bool = False, **kwargs):
        super().__init__()
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        d_model = configs.d_model
        dropout = configs.dropout
        head_dropout = configs.head_dropout
        individual = configs.individual
        patch_len = configs.patch_len
        stride = configs.stride
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(
                c_in, affine=configs.affine, subtract_last=configs.subtract_last
            )

        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = SeriesDecomp(kernel_size)
            self.model_trend = PatchLinear(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                head_dropout=head_dropout,
                individual=individual,
                **kwargs,
            )
            self.model_res = PatchLinear(
                c_in=c_in,
                context_window=context_window,
                target_window=target_window,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                head_dropout=head_dropout,
                individual=individual,
                **kwargs,
            )
        else:
            if configs.model == "PatchLinear":
                self.model = PatchLinear(
                    c_in=c_in,
                    context_window=context_window,
                    target_window=target_window,
                    patch_len=patch_len,
                    stride=stride,
                    d_model=d_model,
                    head_dropout=head_dropout,
                    individual=individual,
                    **kwargs,
                )
        self.target_window = target_window

    def forward(self, x):
        if self.revin:
            x = self.revin_layer(x, "norm")

        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init = res_init.permute(0, 2, 1)
            trend_init = trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
        else:
            x = x.permute(0, 2, 1)
            x = self.model(x)

        x = x[:, 0, :]

        if self.revin:
            if self.revin_layer.affine:
                target_weight = self.revin_layer.affine_weight[0]
                target_bias = self.revin_layer.affine_bias[0]
                x = x - target_bias
                x = x / (target_weight + self.revin_layer.eps**2)

            target_stdev = self.revin_layer.stdev[:, :, 0]
            if self.revin_layer.subtract_last:
                target_last = self.revin_layer.last[:, :, 0]
                x = x * target_stdev + target_last
            else:
                target_mean = self.revin_layer.mean[:, :, 0]
                x = x * target_stdev + target_mean

        return x


BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
START_TIME_MS = int(datetime.datetime(2018, 1, 1).timestamp() * 1000)
INTERVAL = "4h"
LIMIT = 1000
SHIFTS = [0, 4, 8, 12, 16, 20]
DATA_DIR = "data_temp"
REQUEST_TIMEOUT_SEC = 15
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def export_state_dict_to_bin(
    state_dict: Dict[str, torch.Tensor], out_path: Path
) -> None:
    version = 1
    magic = b"A35M"
    dtype_f32 = 1

    items = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    items.sort(key=lambda kv: kv[0])

    with open(out_path, "wb") as f:
        f.write(magic)
        f.write(struct.pack("<I", version))
        f.write(struct.pack("<I", len(items)))

        for name, tensor in items:
            c_name = name.replace(".", "_").replace("/", "_")
            name_bytes = c_name.encode("utf-8")
            if len(name_bytes) > 65535:
                raise ValueError(f"Tensor name too long: {c_name}")

            t = tensor.detach().cpu().to(torch.float32).contiguous()
            np_data = t.numpy()
            dims = list(np_data.shape)
            if len(dims) > 255:
                raise ValueError(f"Too many dims for tensor: {c_name}")
            data_len = int(np_data.size)

            f.write(struct.pack("<H", len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack("<B", len(dims)))
            for d in dims:
                f.write(struct.pack("<I", int(d)))
            f.write(struct.pack("<B", dtype_f32))
            f.write(struct.pack("<I", data_len))
            f.write(np_data.astype(np.float32, copy=False).tobytes(order="C"))


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Configs:
    def __init__(self):
        self.enc_in = 1
        self.seq_len = 365
        self.pred_len = 1
        self.patch_len = 30
        self.stride = 7
        self.padding_patch = "end"
        self.d_model = 16
        self.dropout = 0.0
        self.head_dropout = 0.0
        self.decomposition = True
        self.kernel_size = 25
        self.revin = True
        self.affine = True
        self.subtract_last = False
        self.individual = False
        self.model = "PatchLinear"


def fetch_all_4h_klines(symbol: str) -> List[list]:
    session = requests.Session()
    klines: List[list] = []
    start_time = START_TIME_MS

    while True:
        params = {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": LIMIT,
            "startTime": start_time,
        }

        data = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = session.get(
                    BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT_SEC
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.RequestException as err:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Failed to fetch klines after {MAX_RETRIES} attempts: {err}"
                    ) from err
                time.sleep(1)

        if not isinstance(data, list) or not data:
            break

        klines.extend(data)
        start_time = int(data[-1][0]) + 1

        if len(data) < LIMIT:
            break

    return klines


def to_dataframe(klines: List[list]) -> pd.DataFrame:
    if not klines:
        raise ValueError("No kline data returned from Binance.")

    df = pd.DataFrame(klines).iloc[:, :6]
    df.columns = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna()
        .drop_duplicates(subset=["Datetime"])
        .sort_values("Datetime")
        .reset_index(drop=True)
    )

    now_utc = pd.Timestamp.utcnow().tz_localize(None)
    df = df[df["Datetime"] + pd.Timedelta(hours=4) <= now_utc].copy()

    if df.empty:
        raise ValueError("Kline dataframe is empty after cleaning.")
    return df


def save_shifted_daily_csv(
    df_4h: pd.DataFrame, coin: str, data_dir: str
) -> Dict[int, pd.DataFrame]:
    shifted_frames: Dict[int, pd.DataFrame] = {}

    for offset in SHIFTS:
        shifted = df_4h.copy()
        shifted["Datetime"] = shifted["Datetime"] - pd.Timedelta(hours=offset)

        df_1d = (
            shifted.set_index("Datetime")
            .resample("1D")
            .agg(
                Open=("Open", "first"),
                High=("High", "max"),
                Low=("Low", "min"),
                Close=("Close", "last"),
                Volume=("Volume", "sum"),
                CandleCount=("Close", "count"),
            )
            .dropna()
        )
        df_1d = df_1d[df_1d["CandleCount"] == 6].drop(columns=["CandleCount"])
        df_1d.index = df_1d.index + pd.Timedelta(hours=offset)

        shifted_frames[offset] = df_1d
        logger.info("Shift %02dH generated (%d candles)", offset, len(df_1d))

    return shifted_frames


def create_shifted_data(coin: str) -> Tuple[int, Optional[Dict[int, pd.DataFrame]]]:
    """Create shifted data for a given coin symbol."""
    coin = coin.upper().strip()
    if not coin.isalnum():
        logger.error("Invalid coin symbol: %s", coin)
        return 1, None

    symbol = f"{coin}USDT"
    logger.info("Fetching %s %s klines from Binance...", symbol, INTERVAL)

    try:
        klines = fetch_all_4h_klines(symbol)
        df_4h = to_dataframe(klines)
        shifted_frames = save_shifted_daily_csv(df_4h, coin, DATA_DIR)
    except Exception as err:
        logger.error("Data generation failed for %s: %s", symbol, err)
        return 1, None

    logger.info("Completed data generation for %s.", coin)
    return 0, shifted_frames


def load_financial_data_df(
    df: pd.DataFrame, seq_len: int, features: List[str] = ["Close"]
):
    """Loads and transforms data using log-difference."""
    transformed = []
    for feat in features:
        series = df[feat].astype(np.float32).values
        transformed.append(np.diff(np.log(series)))

    data = np.stack(transformed, axis=1).astype(np.float32)
    X, y = [], []

    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i, 0])

    return np.array(X), np.array(y)


def solve_head_geometrically(model, X_tr, y_tr, device):
    """Refines model head using Ordinary Least Squares (Normal Equations)."""
    model.eval()
    X_tensor = torch.from_numpy(X_tr).to(device)

    with torch.no_grad():
        if model.revin:
            X_tensor = model.revin_layer(X_tensor, "norm")

        if model.decomposition:
            res_init, trend_init = model.decomp_module(X_tensor)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(
                0, 2, 1
            )

            B, C, _ = res_init.shape
            res_f = model.model_res.patch_conv(
                model.model_res.padding_patch_layer(res_init).reshape(B * C, 1, -1)
            )
            trend_f = model.model_trend.patch_conv(
                model.model_trend.padding_patch_layer(trend_init).reshape(B * C, 1, -1)
            )
            feat = torch.cat(
                [res_f.reshape(B, C, -1)[:, 0, :], trend_f.reshape(B, C, -1)[:, 0, :]],
                dim=1,
            )
        else:
            z = X_tensor.permute(0, 2, 1)
            B, C, _ = z.shape
            z = model.model.patch_conv(
                model.model.padding_patch_layer(z).reshape(B * C, 1, -1)
            )
            feat = z.reshape(B, C, -1)[:, 0, :]

    X_mat = feat.cpu().numpy()
    y_vec = y_tr
    X_bias = np.hstack([X_mat, np.ones((X_mat.shape[0], 1), dtype=np.float32)])
    XTX = X_bias.T @ X_bias + 1e-3 * np.eye(X_bias.shape[1])
    XTy = X_bias.T @ y_vec
    w = np.linalg.solve(XTX, XTy)

    return torch.from_numpy(w[:-1]), torch.from_numpy(w[-1:])


def train_lbfgs(model, X_tr, y_tr, criterion, device):
    print("Phase 3: Starting LBFGS Polishing...")
    model.train()
    X = torch.from_numpy(X_tr).to(device)
    y = torch.from_numpy(y_tr).to(device)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        output = model(X).view_as(y)
        loss = criterion(output, y)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f"LBFGS Final Loss: {loss.item():.6f}")
    return loss.item()


def train_sam_epoch(model, loader, criterion, optimizer, device, rho=0.05):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        output = model(X).view_as(y)
        loss = criterion(output, y)
        loss.backward()
        grad_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]
            )
        )
        scale = rho / (grad_norm + 1e-12)
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    e_w = p.grad * scale
                    p.add_(e_w)
                    p.optim_e_w = e_w

        optimizer.zero_grad()
        criterion(model(X).view_as(y), y).backward()

        with torch.no_grad():
            for p in model.parameters():
                if hasattr(p, "optim_e_w"):
                    p.sub_(p.optim_e_w)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_std: float = 0.001,
) -> float:
    model.train()
    total_loss = 0
    for X, y in loader:
        if noise_std > 0:
            X = X + torch.randn_like(X) * noise_std

        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X).view_as(y), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_coin(coin: str, device: torch.device):
    coin = coin.upper()
    train_seed = 42
    _set_global_seed(train_seed)

    save_path = CURRENT_DIR / f"{coin}.pth"

    print(f"\n" + "=" * 50)
    print(f"Starting Training for {coin}")
    print("=" * 50)

    print(f"Creating shifted data for {coin}...")
    status, shifted_frames = create_shifted_data(coin)
    if status != 0 or shifted_frames is None:
        print(f"Failed to create shifted data for {coin}")
        return

    config = Configs()

    phases = [0, 4, 8, 12, 16, 20]
    train_data = []
    for p in phases:
        if p in shifted_frames:
            train_data.append(load_financial_data_df(shifted_frames[p], config.seq_len))

    if not train_data:
        print(f"Error: No data found for {coin}")
        return

    model = Model(config).to(device)

    print(
        f"Phase 1: Calculating average OLS head across {len(train_data)} data shifts..."
    )
    all_w, all_b = [], []
    for X_init, y_init in train_data:
        w, b = solve_head_geometrically(model, X_init, y_init, device)
        all_w.append(w)
        all_b.append(b)

    avg_w = torch.stack(all_w).mean(dim=0)
    avg_b = torch.stack(all_b).mean(dim=0)

    with torch.no_grad():
        if model.decomposition:
            mid = avg_w.shape[0] // 2
            model.model_res.head.linear.weight.copy_(avg_w[:mid].view(1, -1))
            model.model_trend.head.linear.weight.copy_(avg_w[mid:].view(1, -1))
            model.model_res.head.linear.bias.fill_(avg_b.item() / 2.0)
            model.model_trend.head.linear.bias.fill_(avg_b.item() / 2.0)
        else:
            model.model.head.linear.weight.copy_(avg_w.view(1, -1))
            model.model.head.linear.bias.copy_(avg_b)
    print(f"Multi-Phase OLS Init complete (avg_w_mean: {avg_w.mean():.6f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"Training {coin} PatchLinear Model with OLS Warm-start...")

    epochs = 100
    for epoch in range(epochs):
        p_idx = (epoch // 5) % len(train_data)
        X_tr, y_tr = train_data[p_idx]

        loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=64,
            shuffle=True,
        )

        train_loss = train_epoch(model, loader, criterion, optimizer, device)

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"[{coin}] Epoch {epoch+1:03d} | Phase {phases[p_idx]:02d}H | Loss: {train_loss:.6f}"
            )

    print(f"Phase 3: Starting SAM Robustness Training for {coin} (100 steps)...")
    sam_optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    X_sam, y_sam = train_data[0]
    sam_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_sam), torch.from_numpy(y_sam)),
        batch_size=64,
        shuffle=True,
    )

    for e in range(100):
        sam_loss = train_sam_epoch(model, sam_loader, criterion, sam_optimizer, device)
        if (e + 1) % 10 == 0:
            print(f"SAM Epoch {e+1:03d} | Loss: {sam_loss:.6f}")

    X_final, y_final = train_data[0]
    train_lbfgs(model, X_final, y_final, criterion, device)

    state_dict = model.state_dict()
    bin_path = CURRENT_DIR / f"{coin}.bin"
    export_state_dict_to_bin(state_dict, bin_path)
    print(f"Successfully exported {coin}. Binary model saved to: {bin_path.name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = CURRENT_DIR / "cfg.yaml"

    if len(sys.argv) >= 2:
        coins = [sys.argv[1]]
    elif cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            coins = cfg.get("coins", [])
    else:
        print(f"Error: No symbol provided and {cfg_path} not found.")
        print("Usage: python3 train_models.py <SYMBOL>")
        sys.exit(1)

    if not coins:
        print("Error: No coins found in cfg.yaml or command line.")
        sys.exit(1)

    print(f"List of coins to train: {', '.join(coins)}")

    for coin in coins:
        try:
            train_coin(coin, device)
        except Exception as e:
            print(f"Error training {coin}: {e}")
            continue

    print("\nAll training tasks completed.")


if __name__ == "__main__":
    main()
