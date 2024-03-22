_base_ = ['./defeat_r18_gflv1_r101_fpn_coco_1x.py']
model = dict(
    #pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='LDHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        #loss_kd=dict(
        #    type='KnowledgeDistillationKLDivLoss', loss_weight=10, T=2),
        loss_im=dict(type='IMLoss', loss_weight=2.0),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        imitation_method='decouple'  # gibox, finegrain, decouple, fitnet
    ))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)


