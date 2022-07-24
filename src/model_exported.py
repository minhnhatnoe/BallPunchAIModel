# %% [markdown]
# # Import everything
# 

# %%
from typing import Tuple
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from settings import cfg
import export_result
from settings.modules import models

# %% [markdown]
# # Load everything

# %%
config = cfg.TrainConfig(models.get_vgg16(True))


# %% [markdown]
# # Train

# %%


def train(train_idx: np.ndarray) -> 'Tuple(float, float)':
    config.model.train()
    train_dataloader = config.get_dataloader(train_idx)
    total_loss_train = 0

    prediction_array = []
    label_array = []
    for image_temp, label in tqdm(train_dataloader):
        image = image_temp.clone()
        image = image.to(config.device, dtype=torch.float)
        image = config.transforms(image)
        if config.use_grayscale:
            image = config.grayscale(image)
        output = config.model(image)

        label = label.to(config.device, dtype=torch.uint8)
        batch_loss = config.criterion(output, label)
        total_loss_train += batch_loss.item()

        prediction = output.argmax(dim=1)

        prediction_array.append(prediction.cpu().numpy())
        label_array.append(label.cpu().numpy())

        config.optimizer.zero_grad()
        batch_loss.backward()
        config.optimizer.step()
        config.model.zero_grad()

    prediction_array = np.concatenate(prediction_array)
    label_array = np.concatenate(label_array)

    total_accuracy_train = (prediction_array == label_array).sum().item()
    f1_score_train = f1_score(label_array, prediction_array, average='macro')

    return (total_loss_train/train_idx.shape[0],
        total_accuracy_train/train_idx.shape[0],
        f1_score_train)


# %%
def judge(judge_idx: np.ndarray) -> 'Tuple(float, float)':
    config.model.eval()
    judge_dataloader = config.get_dataloader(judge_idx)
    total_loss_judge = 0

    prediction_array = []
    label_array = []
    with torch.no_grad():
        for image_temp, label in tqdm(judge_dataloader):
            image = image_temp.clone()
            image = image.to(config.device, dtype=torch.float)
            if config.use_grayscale:
                image = config.grayscale(image)
            output = config.model(image)
            label = label.to(config.device, dtype=torch.uint8)

            batch_loss = config.criterion(output, label)
            total_loss_judge += batch_loss.item()

            prediction = output.argmax(dim=1)
            prediction_array.append(prediction.cpu().numpy())
            label_array.append(label.cpu().numpy())

    prediction_array = np.concatenate(prediction_array)
    label_array = np.concatenate(label_array)

    total_accuracy_judge = (prediction_array == label_array).sum().item()
    f1_score_judge = f1_score(label_array, prediction_array, average='macro')

    return (total_loss_judge/judge_idx.shape[0],
        total_accuracy_judge/judge_idx.shape[0],
        f1_score_judge)

# %%
min_loss_judge = float('inf')
last_loss_judge = float('inf')
des_sequence = 0
under_min = 0
last_submit = 0

for epoch, (train_idx, judge_idx) in enumerate(config.get_split()):
    print(f'''Starting epoch {epoch+1}
    | Train size:     {train_idx.shape[0]}   | Judge size:     {judge_idx.shape[0]}''')
    avg_loss_train, avg_accu_train, f1_score_train = train(train_idx)
    avg_loss_judge, avg_accu_judge, f1_score_judge = judge(judge_idx)

    print(
        f'''Epoch: {epoch+1} 
    | Train Loss:     {avg_loss_train:.3f}   | Judge Loss:     {avg_loss_judge:.3f}
    | Train Accuracy: {avg_accu_train:.3f}   | Judge Accuracy: {avg_accu_judge:.3f}
    | Train F1 Score: {f1_score_train:.3f}   | Judge F1 Score: {f1_score_judge:.3f}''')

    if last_loss_judge < avg_loss_judge:
        des_sequence += 1
    if min_loss_judge < avg_loss_judge:
        under_min += 1
    else:
        config.save_checkpoint()
        print(f'''Judge loss improved:
    | From:           {min_loss_judge:.3f}   | To: {avg_loss_judge:.3f}''')
        min_loss_judge = avg_loss_judge
        under_min = des_sequence = 0

    if under_min >= cfg.early_stop:
        print(f"Early stop. Not better than best for {under_min} epochs.")
        config.load_best()
        print(f"Best model loaded.")
        des_sequence = under_min = 0
    elif des_sequence >= cfg.des_sequence_early_stop:
        print(f"Early stop. Not improved for {des_sequence} epochs.")
        config.load_best()
        print(f"Best model loaded.")
        des_sequence = under_min = 0

    if last_submit == 0:
        last_submit = f1_score_judge
        print("Initial submit")
        export_result.submit(config)
    elif f1_score_judge - last_submit > 0.05:
        print(f'''Submitting:
    | F1 Score: {f1_score_judge:.3f} | Last Submit: {last_submit:.3f}''')
        last_submit = f1_score_judge
        export_result.submit(config)
    else:
        print(f'''Not submitting:
    | F1 Score: {f1_score_judge:.3f} | Last Submit: {last_submit:.3f}''')
    print("\n____________________________________________")



