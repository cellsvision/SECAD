import pickle
import numpy as np


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(data, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)


def stat_pkl(pkl_data, target_class=None):
    if target_class is None:
        target_class = ['hsil', 'agc']
    for k in target_class:
        print(k, pkl_data['result'][k].shape[0])


def filter_large_box(pkl_data, base_mpp=0.2738, target_class=None, thresh=600):
    if target_class is None:
        target_class = ['hsil', 'agc']
    mpp = pkl_data['mpp']
    rescale_ratio = mpp / base_mpp
    thresh = thresh * rescale_ratio

    result = pkl_data['result']
    for label in target_class:
        if label not in result:
            continue
        cur_arr = result[label].copy()
        if cur_arr.shape[0] == 0:
            continue
        w, h = cur_arr[:, 2] - cur_arr[:, 0], cur_arr[:, 3] - cur_arr[:, 1]
        idx_keep = np.bitwise_and(w < thresh, h < thresh)
        result[label] = result[label][idx_keep]

    pkl_data['result'] = result

    return pkl_data


def pre_func(pkl_data):
    return filter_large_box(pkl_data)


def main():
    data = load_pkl('1903199_2019154.mrxs.pkl')
    stat_pkl(data)
    data = filter_large_box(data)
    stat_pkl(data)
    save_pkl(data, '1903199_2019154_post.mrxs.pkl')


if __name__ == '__main__':
    main()
