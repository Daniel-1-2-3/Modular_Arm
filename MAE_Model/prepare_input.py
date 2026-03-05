import json
import torch
class Prepare:
    def __init__(self, stats_path="MAE_model/rgb_stats.json"):
        with open(stats_path, "r") as f:
            data = json.load(f)
        self.mean = torch.tensor(data["mean_rgb"], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(data["std_rgb"], dtype=torch.float32).view(1, 3, 1, 1)

    def normalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(img.device)
        std = self.std.to(img.device)
        return (img - mean) / std