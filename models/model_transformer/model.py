from utils.registry import Registry
import torch
from models.model_abs import Model
from utils.util import read_pkl_dict_list_by_key
from .common_utils.model_util import get_post_result, TransformerEncoder
from .model_cfg import infer_config
from .common_utils.data_util import load_pkl, get_det_box_feat, get_feature_map_feat
import os

cur_path = os.path.dirname(__file__)


@Registry.models.register(cur_path.split(os.sep)[-1])
class transformer_model(Model):
    def __init__(self, model_name, gpus, **kwargs):
        self.feat_type = kwargs.get('feat_type', 'det_box')     # feat_type is required in config file, default det_box
        config = infer_config[self.feat_type]
        if 'cfg' in kwargs:
            config.update(kwargs['cfg'])
        super().__init__()
        self.model_name = model_name
        self.thresh = config.thresh
        self.feat_config = config.feat_config
        self.device = config.device
        self.model = TransformerEncoder(**config.model_config)
        self.model.load_state_dict(torch.load(os.path.join(cur_path, 'model_zoo', model_name + config.model_suffix)))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        self.model = self.model.to(config.device)
        self.model.eval()

    def infer(self, pkl_path_dict_list):
        pkl_path_list, pkl_data_list = read_pkl_dict_list_by_key(pkl_path_dict_list, self.feat_type)

        infer_results = []
        for pkl_path, pkl_result in zip(pkl_path_list, pkl_data_list):
            feat = self.__preprocess__(pkl_result)
            with torch.no_grad():
                conf = self.model(feat)
            infer_ret = dict(pkl_name=os.path.basename(pkl_path))
            infer_ret.update(self.__postprocess__(conf))
            infer_results.append(infer_ret)

        return infer_results

    def __preprocess__(self, pkl_result):
        if self.feat_type == 'det_box':
            feat = get_det_box_feat(pkl_result, **self.feat_config)
        elif self.feat_type == 'feature_map':
            feat = get_feature_map_feat(pkl_result, **self.feat_config)
        else:
            raise Exception(f'invalid feat type: {self.feat_type}')
        feat = torch.Tensor(feat).unsqueeze(0)  # 1* n_sequences * N_dim
        feat = feat.to(self.device)
        return feat

    def __postprocess__(self, conf):
        return get_post_result(conf, thresh=self.thresh)
