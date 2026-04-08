# Copyright (c) ai4rs. All rights reserved.
from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.two_stage import TwoStageDetector
from torch import Tensor
from mmdet.structures import SampleList

import torch.nn as nn
from .ega import LFE_Module  # 引入你整理好的 EGA 模块

@MODELS.register_module()
class GSDet(TwoStageDetector):
    r"""Implementation of `GSDet`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        # --- 新增：初始化 EGA 增强模块 ---
        # 对应 ResNet50 的 P3(512), P4(1024), P5(2048)
        self.ega_layers = nn.ModuleList([
            LFE_Module(dim=512, stage=0, mlp_ratio=2.0, drop_path=0.1, 
                       act_layer=nn.ReLU, norm_layer=dict(type='BN')), 
            LFE_Module(dim=1024, stage=1, mlp_ratio=2.0, drop_path=0.1, 
                       act_layer=nn.ReLU, norm_layer=dict(type='BN')), 
            LFE_Module(dim=2048, stage=1, mlp_ratio=2.0, drop_path=0.1, 
                       act_layer=nn.ReLU, norm_layer=dict(type='BN'))
        ])

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                raise NotImplementedError
            self.rpn_head = MODELS.build(rpn_head_)
            
    def extract_feat(self, batch_inputs: Tensor) -> tuple:
        #print("--- Debug: Entering extract_feat ---")
        """提取特征并应用 EGA 拦截增强"""
        # 1. 显式调用 backbone，不通过父类接口
        x = self.backbone(batch_inputs)  # 返回 ResNet 的多尺度特征 (P2, P3, P4, P5)
        
        # 2. 转换成 list 准备增强
        enhanced_x = list(x)
        
        # 3. 对 P3, P4, P5 进行增强 (对应 ResNet 输出的索引 1, 2, 3)
        # 注意：i=1 是 P3 (512通道), i=2 是 P4 (1024通道), i=3 是 P5 (2048通道)
        for i in range(1, len(enhanced_x)):
            # ega_layers[0] 对应 P3, [1] 对应 P4, [2] 对应 P5
            ega_idx = i - 1
            if ega_idx < len(self.ega_layers):
                # 核心修正：应用 EGA 增强
                # 建议增加残差连接（+ x[i]）以保证训练初期的稳定性
                enhanced_x[i] = self.ega_layers[ega_idx](enhanced_x[i]) + enhanced_x[i]
        
        x = tuple(enhanced_x)
        
        # 4. 显式传入 Neck (FPN)
        if self.with_neck:
            x = self.neck(x)
            
        return x

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs)

        losses = dict()
        assert self.with_rpn == True
        rpn_losses, results_list = self.rpn_head.loss(
            x, batch_data_samples)

        keys = rpn_losses.keys()
        for key in list(keys):
            if 'loss' in key and 'rpn' not in key:
                rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(x, results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        rpn_results_list = self.rpn_head.predict(
            x, batch_data_samples)

        results_list = self.roi_head.predict(
            x, rpn_results_list=rpn_results_list,
            batch_data_samples=batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)
        rpn_results_list = self.rpn_head.predict(
            x, batch_data_samples)
        roi_outs = self.roi_head.forward(x, rpn_results_list,
                                         batch_data_samples)
        results = results + (roi_outs, )
        return results
