# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='PiNetResNet_CIFARWoutBN',
        depth=14,
        num_stages=2,
        out_indices=(1, ),
        strides=(1, 2),
        dilations=(1, 1),
        style='pytorch'),
    neck=dict(type='GlobalAveragePoolingConv2d'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))
