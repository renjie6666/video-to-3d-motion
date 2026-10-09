"""Inference config for Kineo's public Human3.6M RTMPose-l checkpoint.

Source: https://huggingface.co/charlesjvt/rtmpose-l-h36m-384x288
Matches the author's model and codec; training-only settings are omitted.
Use checkpoints/rtmpose_h36m.pth. Input sizes below are (width, height).
Pass original images and original-pixel bboxes to inference_topdown; the
pipeline crops to this resolution and MMPose restores image coordinates.
The checkpoint supplies its own dataset metadata. Verify left/right leg
channel order against annotations before enabling flip testing.
"""

_base_ = ["../../models/mmpose/_base_/default_runtime.py"]

codec = dict(
    type="SimCCLabel",
    input_size=(288, 384),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=None,
        frozen_stages=4,
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=1024,
        out_channels=17,
        input_size=codec["input_size"],
        in_featuremap_size=tuple(s // 32 for s in codec["input_size"]),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=False),
)

# Only the pipeline is used by inference_topdown; no training dataset is built.
test_pipeline = [
    dict(type="LoadImage", backend_args=dict(backend="local")),
    dict(type="GetBBoxCenterScale", padding=1.25),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# Fallback metainfo for model inspection without a checkpoint. With the
# published checkpoint, MMPose gives its dataset_meta priority over this.
train_dataloader = dict(
    dataset=dict(metainfo=dict(from_file="configs/_base_/datasets/h36m.py"))
)
