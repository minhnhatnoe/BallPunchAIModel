import csv
from os import environ
from datetime import datetime
import torch
from tqdm import tqdm
from helper import loader, boilerplate
from settings.cfg import model, device
from settings.cfg import batch_size, tests_paths
from settings.cfg import model_path, result_path
from settings.cfg import kaggle_path

print("Loading state dict")
model.load_state_dict(torch.load(model_path))
print("Loading data")
test_image, test_names = loader.load_data(tests_paths, mmap_mode='c')
Data = boilerplate.ExportDataset

tests = Data(test_image, test_names)
tests_dataloader = torch.utils.data.DataLoader(tests, batch_size=batch_size)

print("Starting evaluation")
with torch.no_grad(), open(result_path, "w", newline='') as result:
    model.eval()
    result_writer = csv.writer(result)
    result_writer.writerow(["Frame", "Label"])
    for image, names in tqdm(tests_dataloader):
        image = image.to(device, dtype=torch.float)

        output = model(image)
        output = output.argmax(dim=1).cpu().numpy()
        assert(output.shape[0] == len(names))
        
        result_writer.writerows(zip(names, output))

choice = input("Do you want to submit the result? ([y]/n)")
if choice == "y" or choice == "":
    print("Uploading results")
    environ['KAGGLE_CONFIG_DIR'] = kaggle_path
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(
        result_path, f"Submitted at {datetime.now()}", "hsgs-hackathon2022")
else:
    print("Upload aborted")