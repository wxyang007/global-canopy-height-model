import torch
from gchm.models.xception_sentinel2 import xceptionS2_08blocks_256

# info
print(xceptionS2_08blocks_256.__doc__)

# load the model with random initialization
model = xceptionS2_08blocks_256()

# load pre-trained weights
# download weights
model = xceptionS2_08blocks_256(in_channels=15, out_channels=1, model_weights="GLOBAL_GEDI_MODEL_0",
                                returns="variances_exp", download_dir="./trained_models")
# note: if using m1 chip on mac, add "map_location='mps'" in line 371 in "xception_sentinel2.py"


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print ("MPS device not found.")

model_weights = "./trained_models/GLOBAL_GEDI_2019_2020/model_1/FT_Lm_SRCB/checkpoint.pt"

model = xceptionS2_08blocks_256(in_channels=15, out_channels=1,
                                model_weights=model_weights,
                                returns="variances_exp")