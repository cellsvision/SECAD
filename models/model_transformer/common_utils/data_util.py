import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from .post_proc import pre_func
import math


def get_file_lst(path, ext=None):
    file_lst = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if ext is not None:
                if os.path.splitext(file)[-1] in ext:
                    file_lst.append(os.path.join(root, file))
            else:
                file_lst.append(os.path.join(root, file))
    return file_lst


def filter_top_boxes(result, top_all, include_class=None):
    if include_class is None:
        include_class = ['asc_us', 'lsil', 'asc_h', 'hsil', 'agc']
    all_keys = include_class

    box_array = []
    for k in result:
        if k not in all_keys:
            continue
        for box in result[k]:
            box_array.append(list(box) + [k])

    box_array = np.array(box_array)
    if box_array.shape[0]:
        box_array = box_array[box_array[:, 4].argsort()[::-1]]
    else:
        return result

    result_array = box_array[:min(top_all, box_array.shape[0])]

    new_dict = {k: result_array[result_array[:, 5] == k][:, :5].astype(np.float) for k in all_keys}

    return new_dict


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def nms_cls_conf(result, target_class=None, iou_thresh=0.5):
    if target_class is None:
        target_class = ['asc_us', 'lsil', 'asc_h', 'hsil', 'agc']

    box_conf_list = []
    label_list = []
    for cls in target_class:
        cur_arr = result[cls]
        if cur_arr.shape[0]:
            box_conf_list.append(cur_arr)
            label_list.extend([target_class.index(cls)] * cur_arr.shape[0])
    # No target boxes in result dict, return None
    if not label_list:
        return None

    det = np.concatenate(box_conf_list, axis=0)
    label = np.array(label_list)

    x1, y1, x2, y2, scores = det[:, 0], det[:, 1], det[:, 2], det[:, 3], det[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    conf_order = scores.argsort()[::-1]

    # keep boxes in descending order (sorted by confidence)
    # overlap stores the overlapping boxes in the same position
    keep = []
    overlap = []
    while conf_order.size > 0:
        i = conf_order[0]
        keep.append(i)
        xx1, yy1, xx2, yy2 = np.maximum(x1[i], x1[conf_order[1:]]), np.maximum(y1[i], y1[conf_order[1:]]), \
                             np.minimum(x2[i], x2[conf_order[1:]]), np.minimum(y2[i], y2[conf_order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0., yy2 - yy1)
        inter = w * h
        iou = inter / np.minimum(areas[i], areas[conf_order[1:]])   # compute iou based on the minimum area
        idx_overlap = np.where(iou > iou_thresh)[0]
        idx_none_lap = np.where(iou <= iou_thresh)[0]

        idx_overlap = idx_overlap + 1
        cmp_idx = np.append([0], idx_overlap)
        min_xx1, min_yy1, max_xx2, max_yy2 = np.amin(x1[conf_order[cmp_idx]]), np.amin(y1[conf_order[cmp_idx]]), \
                                             np.amax(x2[conf_order[cmp_idx]]), np.amax(y2[conf_order[cmp_idx]])

        det[i, :4] = min_xx1, min_yy1, max_xx2, max_yy2
        overlap.append(conf_order[idx_overlap])
        conf_order = conf_order[idx_none_lap + 1]

    feat_list = []
    for idx_x, idx_y in zip(keep, overlap):
        cur_box_conf = np.concatenate([det[idx_x].reshape(-1, 5), det[idx_y]])
        w_h = np.array([det[idx_x][2] - det[idx_x][0], det[idx_x][3] - det[idx_x][1]])
        n_box = cur_box_conf.shape[0]
        cur_label = np.append(label[idx_x], label[idx_y])
        conf_sum = np.zeros((n_box, len(target_class)))
        conf_sum[list(range(n_box)), cur_label] = cur_box_conf[:, -1]
        box_feat = np.append(w_h, np.sum(conf_sum, axis=0))     # [w, h, conf1, conf2, ..., conf_n_class]
        feat_list.append(box_feat)
    feat_arr = np.stack(feat_list, axis=0)

    return feat_arr


def get_det_box_feat(pkl_data, target_class=None, top_all=1000, n_sequences=400, N_dim=7, pre=False):
    if target_class is None:
        target_class = ['asc_us', 'lsil', 'asc_h', 'hsil', 'agc']
    if pre:
        pkl_data = pre_func(pkl_data)
    top_result = filter_top_boxes(pkl_data['result'], top_all, include_class=target_class)
    nms_feat = nms_cls_conf(top_result, target_class)

    ret_feat = np.zeros((n_sequences, N_dim))
    if nms_feat is None:
        return ret_feat

    if nms_feat.shape[1] != N_dim:
        raise Exception(f'feature dim mismatch: {nms_feat.shape[1]} vs {N_dim}')

    idx = min(n_sequences, nms_feat.shape[0])
    ret_feat[:idx, :] = nms_feat[:idx, :]

    return ret_feat


def get_feature_map_feat(pkl_data, target_class=None, top_all=1000, n_sequences=3000, N_dim=2048, pre=False):
    feat = pkl_data['feature_map']
    # --------- center crop ---------
    wsi_size = int(math.sqrt(n_sequences))
    h, w = feat.shape[:2]
    if h <= wsi_size:
        c_h, c_w = h, min(n_sequences // h, w)
        h_start = 0
        w_start = (w - c_w) // 2
    elif w <= wsi_size:
        c_h, c_w = min(n_sequences // w, h), w
        h_start = (h - c_h) // 2
        w_start = 0
    else:
        c_h, c_w = wsi_size, wsi_size
        h_start = (h - c_h) // 2
        w_start = (w - c_w) // 2
    feat = feat[h_start: h_start + c_h, w_start: w_start + c_w]
    # --------- center crop ---------

    feat = feat.reshape((-1, N_dim))

    ret_feat = np.zeros((n_sequences, N_dim))
    idx = min(n_sequences, feat.shape[0])
    ret_feat[:idx, :] = feat[:idx, :]

    return ret_feat


class CervixDataset(Dataset):
    def __init__(self, pkl_dir, csv_path, cls_class=None, feat_config_dict=None):
        super(CervixDataset, self).__init__()
        if cls_class is None:
            cls_class = ['nilm', 'asc_us', 'lsil', 'hsil', 'agc']
        self.cls_class = cls_class
        self.pkl_dir = pkl_dir
        self.data_info = pd.read_csv(csv_path)
        print(dict(pd.Series(self.data_info.iloc[:, 1]).value_counts()))
        self.feat_config_dict = feat_config_dict

    def __len__(self):
        return self.data_info.shape[0]

    def __getitem__(self, idx):
        img_name, label = self.data_info.iloc[idx, :2]
        label = self.cls_class.index(label.lower())
        pkl_path = os.path.join(self.pkl_dir, img_name + '.pkl')
        pkl_data = load_pkl(pkl_path)
        feat = get_pkl_feat(pkl_data, **self.feat_config_dict)
        return torch.Tensor(feat), torch.tensor(label).long()


def get_sample_weights(data_info, weight_dict=None):
    # weight dicts
    # {'truth_code':
    #  {'nilm': num1, 'asc_us': num2, 'lsil': num3, 'hsil': num4, 'agc': num5},
    # 'hospital':
    #  {'zs2y': num1, 'gy3y': num2, 'sfy': num3, 'stzx': num4, 'fsfy': num5, ...},
    #  ...
    # }
    if weight_dict is None:
        weight_dict = {}

    df = data_info.copy()
    df['weight'] = 1.0
    df['truth_code'] = df['truth_code'].apply(lambda x: x.lower())

    if 'truth_code' in weight_dict:
        weights = weight_dict['truth_code']
        for k in weights:
            df.loc[df['truth_code'] == k, 'weight'] = df.loc[df['truth_code'] == k, 'weight'] * weights[k]

    if 'hospital' in weight_dict:
        weights = weight_dict['hospital']
        for k in weights:
            df.loc[df['hospital'] == k, 'weight'] = df.loc[df['hospital'] == k, 'weight'] * weights[k]

    if 'scanner' in weight_dict:
        weights = weight_dict['scanner']
        for k in weights:
            df.loc[df['scanner'] == k, 'weight'] = df.loc[df['scanner'] == k, 'weight'] * weights[k]

    return list(df.weight)


def get_balance_weights(data_info, class_order=None):
    if class_order is None:
        class_order = ['nilm', 'asc_us', 'lsil', 'hsil', 'agc']
    target = np.array([class_order.index(x.lower()) for x in data_info['truth_code']])

    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    return samples_weight


def main():
    data = load_pkl('../../051118-20200801010228.TMAP.pkl')
    ret = nms_cls_conf(data['result'])
    print(ret)


if __name__ == '__main__':
    main()
