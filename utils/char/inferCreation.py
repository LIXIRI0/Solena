import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import config_tiny

ckpt = torch.load(config_tiny.CHECKPOINT_PATH, map_location="cpu")

state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

out_path = config_tiny.CHECKPOINT_PATH.replace(".pth", "_infer.pth")
torch.save(state_dict, out_path)

print(out_path)