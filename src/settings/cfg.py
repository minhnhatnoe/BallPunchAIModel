from settings.modules import models, trans, loss, dev, data, optim, kf

device = dev.device
model = models.get_vgg11(True)
transforms = trans.get_random_transforms()
criterion = loss.get_cross_entropy_loss()
optimizer = optim.get_adam(model.parameters())
idx_gen = kf.get_kfold_class(split_count=10, loop_count=2)

train_paths = data.train_data
tests_paths = data.tests_data
model_path = data.model_path
result_path = data.result_path

import_batch = 10000
test_size = 0.2
batch_size = 64
