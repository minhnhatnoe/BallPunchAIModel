import torch
import config

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")

model = config.get_model()
model = model.to(device)
model.load_state_dict(config.model_path)


