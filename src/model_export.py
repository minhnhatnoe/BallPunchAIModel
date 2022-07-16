import torch
import numpy as np
import config

use_cuda = torch.cuda.is_available()
if not use_cuda:
    print("CUDA not used!")
device = torch.device("cuda" if use_cuda else "cpu")

model = config.get_model()
model = model.to(device)
model.load_state_dict(config.model_path)

test_image = np.load(config.t_path)
test_names = np.load(config.n_path)


