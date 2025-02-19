_base_ = ['./segformer_mit-b0_512x512_40k_part_imagenet.py']

# model settings
model = dict(
    pretrained='models/segformer/pretrain/mit_b2.pth',
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))
