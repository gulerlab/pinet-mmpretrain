_base_ = [
    '../_base_/models/pinet/pinet_shallow_cifar10.py', '../_base_/datasets/cifar10_bs16.py',
    '../_base_/schedules/pinet_resnet14_cifar10_wout_bn_schedule.py', '../_base_/default_runtime.py'
]
