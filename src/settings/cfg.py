from modules import models, trans, loss, dev, data, optim, kf

__all__ = ["device", "model", "transforms", "criterion", "optimizer",
    "kfold", "idx_gen", "data_paths", "result_paths", "import_batch",
    "test_size", "batch_size"]

device = dev.device
model = models.get_vgg11(True)
transforms = trans.get_random_transforms()
criterion = loss.get_cross_entropy_loss()
optimizer = optim.get_adam()(model.parameters())
idx_gen = kf.get_kfold_class(split_count=10, loop_count=2)

data_paths = data.Input
result_paths = data.Output

import_batch = 10000
test_size = 0.2
batch_size = 64
