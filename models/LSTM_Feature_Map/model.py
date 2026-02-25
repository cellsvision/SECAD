import os
import pickle as pickle

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from einops import rearrange

from models.model_abs import Model
from utils.registry import Registry
from utils.util import read_pkl_dict_list_by_key
from .infer_config import cfg


cur_path =os.path.dirname(__file__)
@Registry.models.register(cur_path.split(os.sep)[-1])  # 注册名字需与模型目录名一致
class Cervical_Slide_Classification(Model):
    def __init__(self, gpus,model_name, **kwargs):

        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_list = cfg.label_list

        # Load from config
        self.repeat_time = cfg.repeat_time

        if cfg.model_name == 'lstm':

            from .models.lstm import LSTM
            self.model = LSTM(num_class=len(self.label_list), need_embedding=True, embedding_dim=cfg.embedding_dim, hidden_dim=512)
        else:
            from .models.bilstm_attention import BiLSTM_Attention

            self.model = BiLSTM_Attention(num_class=len(self.label_list), need_embedding=True, embedding_dim=cfg.embedding_dim,
                                          hidden_dim=512)
        model_path = cfg.model_ckpt[model_name]['ckpt_path']
        model_path = os.path.join(cur_path,model_path)
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

    def infer(self, pkl_path_dict_list):
        '''

        :param pkl_path: pkl路径
        :return: 预测类别字符串，每个类别的概率nparray，最大概率值
        '''
        # Predict Slide
        pkl_path_list, pkl_data_list = read_pkl_dict_list_by_key(pkl_path_dict_list, 'feature_map')
        feature_maps = []
        for pkl_path in pkl_path_list:
            feature_map = self.__preprocess__(pkl_path)
            feature_maps.append(feature_map)
        feature_maps = torch.cat(feature_maps, dim=0)
        with torch.no_grad():
            logits = self.model(feature_maps)
            confs = nn.Softmax(dim=1)(logits).cpu().numpy()

        pred_idxs = np.argmax(confs, axis=1)
        preds = [self.label_list[idx] for idx in pred_idxs]
        max_confs = [confs[i][idx] for i, idx in enumerate(pred_idxs)]
        infer_results = []
        for pkl_path, pred, max_conf, conf in zip(pkl_path_list, preds, max_confs, confs):
            pkl_name = os.path.basename(pkl_path)
            result_dict = {self.label_list[i]: conf[i] for i in range(len(conf))}

            result_dict['pkl_name'] = pkl_name
            result_dict['pred'] = pred
            result_dict['ori_pred'] = pred
            result_dict['max_conf'] = max_conf

            infer_results.append(result_dict)
        return infer_results

    def __preprocess__(self, pkl_path, img_size=(50, 50)):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            feature_map = data['feature_map'].astype(np.float32)
            # loc_map = np.array(data['loc_map'])
            # self.patch_size = data['patch_size']
        if len(feature_map.shape) != 3:
            h, w = feature_map.shape[:2]
            feature_map = np.reshape(feature_map, (h, w, -1))
        h, w, c = feature_map.shape
        if h < img_size[0] or w < img_size[1]:
            new_feature_map = np.zeros(shape=(max(img_size[0], h), max(img_size[1], w), c)).astype(np.float32)
            new_feature_map[:h, :w] = feature_map
            feature_map = new_feature_map
        transform_list = [
            # A.Flip(p=1),
            A.CenterCrop(height=img_size[0], width=img_size[1]),
            ToTensorV2()
        ]

        composed_transforms = A.Compose(transform_list)
        transformed = composed_transforms(image=feature_map)
        feature_map = transformed['image']
        feature_map = F.normalize(feature_map, p=2, dim=0)
        if self.repeat_time == 1:

            feature_map = feature_map.unsqueeze(0)
        else:
            feature_map_list = [feature_map]
            for i in range(1, self.repeat_time):
                c, h, w = feature_map.shape
                new_feature_map = rearrange(feature_map, 'c h w -> c (h w)')
                new_feature_map = new_feature_map[:, torch.randperm(new_feature_map.size(1))]
                new_feature_map = rearrange(new_feature_map, 'c (h w) -> c h w', h=h, w=w)
                feature_map_list.append(new_feature_map)
            feature_map = torch.stack(feature_map_list, dim=0)

        return feature_map.to(self.device)


if __name__ == '__main__':
    # infer_list(r'/datasets/Breast/DICOM/fengjiahao/cervix/models/InferCervixTF/pkl_zs8y_20210622_200')
    # infer_list(r'/datasets/Breast/DICOM/fengjiahao/cervix/data/pkl_det_train_16/')

    model = Model(gpus='1,2,3')
    pkl_path_dict_list = [
        {'det_box': 'XXX/sample1.pkl', 'feature_map': '/datasets/Breast/DICOM/fengjiahao/cervix/data/pkl_det_train_16/T2008431.kfb.pkl'},
        {'det_box': 'XXX/sample2.pkl', 'feature_map': '/datasets/Breast/DICOM/fengjiahao/cervix/data/pkl_det_train_16/C201904773_yan.kfb.pkl'},
        {'det_box': 'XXX/sample2.pkl', 'feature_map': '/datasets/Breast/DICOM/fengjiahao/cervix/data/pkl_det_train_16/C201909507.TMAP.pkl'},
    ]
    result = model.infer(pkl_path_dict_list)
    print(result)
