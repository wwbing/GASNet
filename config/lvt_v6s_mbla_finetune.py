# YOLOv6l model
model = dict(
    type='YOLOv6s_mbla',
    pretrained=r"C:\Users\jiahao\Desktop\JIAHAO\paper_with_code\project\Gasnet\weights\emlvt_ds_gd_neck_convsilu_coco\best_ckpt.pt",
    depth_multiple=0.5,
    width_multiple=0.5,
    backbone=dict(
        type='e_lvt_ds',
        ),
    neck=dict(
        type='GD_CSPRepBiFPANNeck',
        
        csp_e=float(1)/2,
        stage_block_type="MBLABlock",
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        anchors_init=[[10,13, 19,19, 33,23],
                      [30,61, 59,59, 59,119],
                      [116,90, 185,185, 373,326]],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=0,
        iou_type='giou',
        use_dfl=True,
        reg_max=16, #if use_dfl is False, please set reg_max to 0
        distill_weight={
            'class': 2.0,
            'dfl': 1.0,
        },
    )
)

solver=dict(
    optim='AdamW',
    lr_scheduler='Cosine',
    lr0=1e-4,
    lrf=0.025,
    momentum=0.9,
    weight_decay=0.05,
    warmup_epochs=2.0,
    warmup_momentum=0.5,
    warmup_bias_lr=1e-6
)

data_aug = dict(
    hsv_h=0.0138,
    hsv_s=0.664,
    hsv_v=0.464,
    degrees=0.373,
    translate=0.245,
    scale=0.898,
    shear=0.602,
    flipud=0.00856,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,
)


training_mode = "conv_silu"
