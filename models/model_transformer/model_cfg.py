from easydict import EasyDict as edict

box_infer_config = edict(
    feat_config=edict(
        target_class=None, top_all=3000, n_sequences=1500, N_dim=7, pre=False
    ),
    model_config=edict(
        n_sequences=1500,
        raw_dim=7,
        d_model=16,
        dim_feedforward=128,
        n_class=5,
        pool='mean',
        pos_encoding=False
    ),
    model_suffix='.pth',
    thresh=0,
    device='cuda',
    gpu='0',    # False for cpu inference
)

feat_infer_config = edict(
    feat_config=edict(
        target_class=None, top_all=None, n_sequences=3025, N_dim=2048, pre=False
    ),
    model_config=edict(
        n_sequences=3025,
        raw_dim=2048,
        d_model=256,
        dim_feedforward=256,
        n_class=5,
        cls_token=False,
        pool='mean',
        pos_encoding=False,
    ),
    model_suffix='.pth',
    thresh=0,
    device='cuda',
    gpu='0',    # False for cpu inference
)

infer_config = edict(
    det_box=box_infer_config,
    feature_map=feat_infer_config
)
