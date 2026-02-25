import sys
sys.path.append(r'../')
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.FeatureMapDataset import FeatureMapDataset
import os
import warnings
import pandas as pd
from tqdm import tqdm
from utils.metrics import plot_testing_results

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

data_csv_path = '' # path to csv file with gt
pkl_root = '' # path to pkl dir
model_path = '' # path to model checkpoint at wsi level

test_data = FeatureMapDataset(data_csv_path,pkl_root, is_train=False, normalize=False,
                                        need_random=False, skip_header=True)


csv_path = '' # result csv saving path


from models.bilstm_attention import BiLSTM_Attention

model = BiLSTM_Attention(num_class=len(test_data.classes_names), need_embedding=True, embedding_dim=1024,
                         hidden_dim=512)
test_data_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=18,
                              pin_memory=True) 
model.load_state_dict(torch.load(model_path)) 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
model = nn.DataParallel(model)
model = model.to(device)


with torch.no_grad():
    filename_list = []
    gt_list = []
    pred_result_list = []
    conf_list = []
    max_conf_list = []

    for i, testing_data in enumerate(tqdm(test_data_loader)):
        imgs, labels, id, hospital = testing_data

        filename_list += list(id)
        gt_list+=list(labels.numpy())

        imgs = imgs.to(device)
        labels = labels.to(device)

        output = model(imgs)
        output = torch.softmax(output.data, dim=1)
        max_conf,pred = torch.max(output, dim=1)

        pred_result_list += list(pred.cpu().numpy())
        conf_list+=list(output.cpu().numpy())
        max_conf_list += list(max_conf.cpu().numpy())
    if len(test_data.classes_names)<=3:
        cm,  avg_f1, weighted_f1,m_score = plot_testing_results(
        gt_list, pred_result_list, [i for i in range(3)], test_data.classes_names)
    else:
        cm,  avg_f1, weighted_f1,m_score = plot_testing_results(
        gt_list, pred_result_list, [i for i in range(len(test_data.classes_names))], test_data.classes_names)

gt_mapping_dict = {
    0:'YIN',
    1:'YANG',
}

result_data = []
for i,(filename,gt,pred_result,conf,max_conf) in enumerate(zip(filename_list,gt_list,pred_result_list,conf_list,max_conf_list)):
    row = [filename]
    row.append(gt_mapping_dict[gt])
    row.append(gt_mapping_dict[pred_result])
    row+=list(conf)
    row.append(max_conf)
    result_data.append(row)

result_data = pd.DataFrame(result_data,columns=['filename','gt','pred_result','YIN_conf','YANG_conf','max_conf'])

result_data.to_csv(csv_path,index=False)








