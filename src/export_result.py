import csv
from os import environ
from datetime import datetime
import torch
from helper import loader, boilerplate
from settings.cfg import tests_paths, result_path, kaggle_path, TrainConfig

def submit(config: TrainConfig) -> None:
    ...

if __name__ == '__main__':
    from tqdm import tqdm
    submit(TrainConfig())
else:
    from tqdm.notebook import tqdm

def submit(config: TrainConfig) -> None:
    test_image, test_names = loader.load_data(tests_paths, mmap_mode='c')
    Data = boilerplate.ExportDataset
    tests = Data(test_image, test_names)
    tests_dataloader = torch.utils.data.DataLoader(
        tests, batch_size=config.batch_size)

    print("Starting evaluation")
    with torch.no_grad(), open(result_path, "w", newline='') as result:
        config.model.eval()
        result_writer = csv.writer(result)
        result_writer.writerow(["Frame", "Label"])
        for image, names in tqdm(tests_dataloader):
            image = image.to(config.device, dtype=torch.float)

            output = config.model(image)
            output = output.argmax(dim=1).cpu().numpy()
            assert(output.shape[0] == len(names))
            
            result_writer.writerows(zip(names, output))

    print("Uploading results")
    environ['KAGGLE_CONFIG_DIR'] = kaggle_path
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.competition_submit(
        result_path, f"Submitted at {datetime.now()}", "hsgs-hackathon2022")
