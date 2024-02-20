# batch normalization
python tools/analysis_tools/analyze_logs.py plot_curve\
 work_dirs/resnet18_8xb16_cifar10/20240215_004552/vis_data/scalars.json\
 work_dirs/pinet_resnet18_8xb16_cifar10/20240215_155754/vis_data/scalars.json\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn/20240216_000221/vis_data/scalars_until_200.json\
 --keys loss\
 --legend "conventional resnet-18" "pinet resnet-18" "pinet resnet-18 wout bn"\
 --title "Effect of Batch Normalization on PiNet Performance-Loss Curves"\
 --out ./work_dirs/022024_bn_comparison_loss_curve.png

python tools/analysis_tools/analyze_logs.py plot_curve\
 work_dirs/resnet18_8xb16_cifar10/20240215_004552/vis_data/scalars.json\
 work_dirs/pinet_resnet18_8xb16_cifar10/20240215_155754/vis_data/scalars.json\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn/20240216_000221/vis_data/scalars_until_200.json\
 --keys accuracy/top1\
 --legend "conventional resnet-18" "pinet resnet-18" "pinet resnet-18 wout bn"\
 --title "Effect of Batch Normalization on PiNet Performance-Accuracy Curves"\
 --out ./work_dirs/022024_bn_comparison_accuracy_curve.png

# pooling layers
python tools/analysis_tools/analyze_logs.py plot_curve\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn/20240216_000221/vis_data/scalars.json\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn_wout_pooling/20240216_202603/vis_data/scalars_until_500.json\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn_conv_pooling/20240219_174927/vis_data/scalars.json\
 --keys loss\
 --legend "pinet resnet-14 with pooling" "pinet resnet-14 wout pooling" "pinet resnet-14 with truncation"\
 --title "Effect of Pooling on PiNet Performance-Loss Curves"\
 --out ./work_dirs/022024_pooling_comparison_loss_curve.png

python tools/analysis_tools/analyze_logs.py plot_curve\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn/20240216_000221/vis_data/scalars.json\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn_wout_pooling/20240216_202603/vis_data/scalars_until_500.json\
 work_dirs/pinet_resnet14_8xb16_cifar10_wout_bn_conv_pooling/20240219_174927/vis_data/scalars.json\
 --keys accuracy/top1\
 --legend "pinet resnet-14 with pooling" "pinet resnet-14 wout pooling" "pinet resnet-14 with truncation"\
 --title "Effect of Pooling on PiNet Performance-Accuracy Curves"\
 --out ./work_dirs/022024_pooling_comparison_accuracy_curve.png

 # shallow architectures
python tools/analysis_tools/analyze_logs.py plot_curve\
 work_dirs/pinet_shallow_cifar10/20240219_185926/vis_data/scalars.json\
 work_dirs/pinet_shallow_two_layer_cifar10_v2/20240220_105144/vis_data/scalars.json\
 --keys loss\
 --legend "pinet shallow with 4 conv" "pinet shallow with 2 conv"\
 --title "Shallow Architecture Experiments-Loss Curves"\
 --out ./work_dirs/022024_shallow_loss_curve.png

python tools/analysis_tools/analyze_logs.py plot_curve\
 work_dirs/pinet_shallow_cifar10/20240219_185926/vis_data/scalars.json\
 work_dirs/pinet_shallow_two_layer_cifar10_v2/20240220_105144/vis_data/scalars.json\
 --keys accuracy/top1\
 --legend "pinet shallow with 4 conv" "pinet shallow with 2 conv"\
 --title "Shallow Architecture Experiments-Accuracy Curves"\
 --out ./work_dirs/022024_shallow_accuracy_curve.png