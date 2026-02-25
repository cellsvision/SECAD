from easydict import EasyDict as edict

cfg = edict()
cfg.model_name = 'att_lstm'
cfg.model_ckpt = {
    'SELSTM20211222': {
        'ckpt_path': r'checkpoints/save/att_lstm_20211222_1_fold0_no_pre/ckpt_101_wf1_0.929.pth',
        'det_model': 'EfficientNet'}
}
cfg.repeat_time = 1
cfg.label_list = ['NEG','POS']
cfg.embedding_dim =1408
