_base_ = './deeplabv3_r50-d8_512x512_4x4_40k_part_imagenet.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
