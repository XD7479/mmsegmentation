_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/part_imagenet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(decode_head=dict(num_classes=40))
