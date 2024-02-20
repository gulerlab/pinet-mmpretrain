auto_scale_lr = dict(base_batch_size=128)
data_preprocessor = dict(
    mean=[
        125.307,
        122.961,
        113.8575,
    ],
    num_classes=10,
    std=[
        51.5865,
        50.847,
        51.255,
    ],
    to_rgb=False)
dataset_type = 'CIFAR10'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(type='PiNetShallow_CIFAR'),
    head=dict(
        in_channels=256,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=10,
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePoolingConv2d'),
    type='ImageClassifier')
optim_wrapper = dict(optimizer=dict(lr=0.0001, type='SGD'))
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        200,
        400,
    ], type='MultiStepLR')
randomness = dict(deterministic=False, seed=None)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar10/',
        pipeline=[
            dict(type='PackInputs'),
        ],
        split='test',
        type='CIFAR10'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(topk=(1, ), type='Accuracy')
test_pipeline = [
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=1)
train_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar10',
        pipeline=[
            dict(crop_size=32, padding=4, type='RandomCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='CIFAR10'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(crop_size=32, padding=4, type='RandomCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=16,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/cifar10/',
        pipeline=[
            dict(type='PackInputs'),
        ],
        split='test',
        type='CIFAR10'),
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(topk=(1, ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/pinet_shallow_cifar10'
