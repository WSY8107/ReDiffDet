angle_version = 'le90'
auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.GSDet_baseline.gsdet',
    ])
data_root = 'data/split_ss_fair1m1.0/'
dataset_type = 'FAIRDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
default_scope = 'mmrotate'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
hbox2hbox = 4
hbox2rbox = 1
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        boxtype2tensor=False,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='mmdet.DetDataPreprocessor'),
    init_cfg=dict(
        checkpoint=
        'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/pretrain/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_new2.pth',
        type='Pretrained'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='mmdet.FPN'),
    roi_head=dict(
        angle_version='le90',
        bbox_head=[
            dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                angle_version='le90',
                bbox_coder=dict(
                    angle_version='le90',
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                        0.1,
                    ],
                    type='MidpointOffsetCoderv2',
                    use_box_type=False),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                dim_feedforward=2048,
                dropout=0.0,
                dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                feat_channels=256,
                loss_bbox=dict(loss_weight=2.0, type='mmdet.L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='mmdet.FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(
                    loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
                num_classes=37,
                num_cls_convs=1,
                num_heads=8,
                num_reg_convs=3,
                pooler_resolution=7,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                type='Hbox2RboxLayer'),
            dict(
                act_cfg=dict(inplace=True, type='ReLU'),
                angle_version='le90',
                bbox_coder=dict(
                    angle_version='le90',
                    edge_swap=True,
                    norm_factor=None,
                    proj_xy=True,
                    target_means=(
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ),
                    target_stds=(
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ),
                    type='DeltaXYWHTRBBoxCoder',
                    use_box_type=False),
                cls_predictor_cfg=dict(type='mmdet.Linear'),
                dim_feedforward=2048,
                dropout=0.0,
                dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                feat_channels=256,
                loss_bbox=dict(loss_weight=2.0, type='mmdet.L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='mmdet.FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(
                    loss_weight=5.0, mode='linear', type='RotatedIoULoss'),
                num_classes=37,
                num_cls_convs=1,
                num_heads=8,
                num_reg_convs=3,
                pooler_resolution=7,
                reg_predictor_cfg=dict(type='mmdet.Linear'),
                type='Rbox2RboxLayer'),
        ],
        bbox_roi_extractor=[
            dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='mmdet.SingleRoIExtractor'),
            dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    clockwise=True,
                    out_size=7,
                    sample_num=2,
                    type='RoIAlignRotated'),
                type='RotatedSingleRoIExtractor'),
        ],
        num_proposals=900,
        num_stages=2,
        stage_loss_weights=[
            1,
            1,
        ],
        type='Decoder'),
    rpn_head=dict(
        criterion=dict(
            assigner=dict(
                candidate_topk=5,
                center_radius=2.5,
                match_costs=[
                    dict(
                        alpha=0.25,
                        eps=1e-08,
                        gamma=2.0,
                        type='mmdet.FocalLossCost',
                        weight=2.0),
                    dict(
                        box_format='xyxy', type='mmdet.BBoxL1Cost',
                        weight=5.0),
                    dict(iou_mode='giou', type='mmdet.IoUCost', weight=2.0),
                ],
                type='Hbox2HboxLayerMatcher'),
            loss_bbox=dict(
                loss_weight=5.0, reduction='sum', type='mmdet.L1Loss'),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=2.0,
                reduction='sum',
                type='mmdet.FocalLoss',
                use_sigmoid=True),
            loss_giou=dict(
                loss_weight=2.0, reduction='sum', type='mmdet.GIoULoss'),
            num_classes=37,
            type='Hbox2HboxLayerCriterion'),
        deep_supervision=True,
        feat_channels=256,
        num_classes=37,
        num_heads=4,
        num_proposals=900,
        prior_prob=0.01,
        roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='mmdet.SingleRoIExtractor'),
        single_head=dict(
            act_cfg=dict(inplace=True, type='ReLU'),
            dim_feedforward=2048,
            dropout=0.0,
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
            num_cls_convs=1,
            num_heads=8,
            num_reg_convs=3,
            type='SingleHbox2HboxHead'),
        type='Hbox2HboxLayer'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=900,
            nms=dict(iou_threshold=0.6, type='nms_rotated'),
            use_nms=True),
        rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    iou_cost=dict(
                        iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    reg_cost=dict(
                        angle_version='le90',
                        box_format='xywht',
                        type='RBBoxL1Cost',
                        weight=2.0),
                    topk=2,
                    type='TopkHungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='mmdet.PseudoSampler')),
            dict(
                assigner=dict(
                    cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                    iou_cost=dict(
                        iou_mode='iou', type='RotatedIoUCost', weight=5.0),
                    reg_cost=dict(
                        angle_version='le90',
                        box_format='xywht',
                        type='RBBoxL1Cost',
                        weight=2.0),
                    topk=2,
                    type='TopkHungarianAssigner'),
                pos_weight=1,
                sampler=dict(type='mmdet.PseudoSampler')),
        ],
        rpn=None),
    type='GSDet')
num_classes = 37
num_proposals = 900
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=5e-06, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
pretrain = 'https://www.modelscope.cn/models/wokaikaixinxin/ai4rs/resolve/master/GSDet_baseline/pretrain/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_new2.pth'
rbox2rbox = 1
resume = True
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='test/images/'),
        data_root='data/split_ss_fair1m1.0/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='FAIRDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    format_only=True,
    merge_patches=True,
    outfile_prefix='./work_dirs/fair1m1.0/Task1',
    type='FAIRMetric')
test_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=6)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=2,
    dataset=dict(
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        data_root='data/split_ss_fair1m1.0/',
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                direction=[
                    'horizontal',
                    'vertical',
                    'diagonal',
                ],
                prob=0.75,
                type='mmdet.RandomFlip'),
            dict(type='mmdet.PackDetInputs'),
        ],
        type='FAIRDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(
        direction=[
            'horizontal',
            'vertical',
            'diagonal',
        ],
        prob=0.75,
        type='mmdet.RandomFlip'),
    dict(type='mmdet.PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        data_root='data/split_ss_fair1m1.0/',
        pipeline=[
            dict(backend_args=None, type='mmdet.LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='mmdet.Resize'),
            dict(
                box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
            dict(
                box_type_mapping=dict(gt_bboxes='rbox'),
                type='ConvertBoxType'),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='mmdet.PackDetInputs'),
        ],
        test_mode=True,
        type='FAIRDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(metric='mAP', type='FAIRMetric')
val_pipeline = [
    dict(backend_args=None, type='mmdet.LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='mmdet.Resize'),
    dict(box_type='qbox', type='mmdet.LoadAnnotations', with_bbox=True),
    dict(box_type_mapping=dict(gt_bboxes='rbox'), type='ConvertBoxType'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='mmdet.PackDetInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='RotLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/gsdet_fair1m_stable_v2'
