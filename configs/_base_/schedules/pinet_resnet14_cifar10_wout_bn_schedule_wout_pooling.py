# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.00000001))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[200, 600], gamma=0.1)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=1000, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=128)
