_base_ = [
    '../_base_/models/pinet/pinet_resnet14_cifar10_wout_bn_wout_pooling.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/pinet_resnet14_cifar10_wout_bn_schedule_wout_pooling_constant_lr.py', '../_base_/default_runtime.py'
]
