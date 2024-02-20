# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='PiNetShallowTwoLayer_CIFAR'),
    neck=dict(type='GlobalAveragePoolingConv2d'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
