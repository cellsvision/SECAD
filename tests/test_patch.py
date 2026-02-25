import sys
sys.path.append(r'../')
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from datasets.PatchDataset import PatchDataset
import os
import warnings
import pandas as pd
from tqdm import tqdm
from utils.metrics import plot_testing_results

os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
# val_data =r'/datasets/fengjiahao/SerousEffusion/data20230630/test/'
# xml_root = r'/datasets/fengjiahao/SerousEffusion/data20230630/cropped_patch_xml/'
val_data =r'/public/tmp/fengjiahao/serous_effusion/data20231129/val/'
xml_root = r'/public/tmp/fengjiahao/serous_effusion/data20231129/cropped_patch_xml/'
image_size=(640,640)
val_data = PatchDataset(val_data, xml_root=xml_root, img_size=image_size, is_train=False)


csv_path = r'datasets/csv/serous_effusion_data_path_20231108.csv'




from utils.model_util import get_model
from utils.trainer2 import Trainer
model_name = r'convnext'
model = get_model(model_name, need_head=True, pretrained=True, num_classes=len(val_data.classes_names), )

test_data_loader = DataLoader(val_data, batch_size=128, shuffle=False, num_workers=18,
                              pin_memory=True)  # 使用DataLoader加载数据
# state_dict = torch.load(r'../save/20230701//convnext_820_820_ce_out_box_adc_fix/convnext_820_820_ce_out_box_adc_fix_ckpt_35.pth')
state_dict = torch.load(r'../save/20231129/convnext_640_640_ce_out_box_adc/convnext_640_640_ce_out_box_adc_ckpt_25.pth')
model.load_state_dict(state_dict) #final  科研V2 4分类


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.eval()
# model = nn.DataParallel(model)
# model = model.to(device)
from models.losses.multi_cls_weight_loss import Multi_Class_Weight_Loss

matrix = [
    [0, 1, 3],
    [2, 0, 0.5],
    [3, 1, 0],
]
criterion = Multi_Class_Weight_Loss(class_num=len(val_data.classes_names), alpha_matrix=matrix)
train_task = 'softmax'
trainer = Trainer(model, criterion=criterion, classes_names=val_data.classes_names, optimizer=None,
                  writer=None, train_task=train_task)

val_loss, acc, avg_precision, avg_recall, avg_f1, weighted_f1, auc = trainer.validate(
    test_data_loader, 0)



