import torch
from torch.utils.data import DataLoader
import argparse
import torch_optimizer as optim
from tensorboardX import SummaryWriter
from utils.trainer_wsi2 import Trainer
import torch.nn as nn
from datasets.FeatureMapDataset import FeatureMapDataset
import os
from torch.utils.data.sampler import WeightedRandomSampler
import warnings
from utils.DataLoaderX import DataLoaderX

warnings.filterwarnings("ignore")
import random
random.seed(666)
if __name__ == "__main__":
    # python -m visdom.server
    # mp.set_start_method('spawn')

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # parser.add_argument("--pretrained_weights", type=str, default="", help="if specified starts from checkpoint model")

    parser.add_argument("--epoch", type=int, default=800, help="epoch")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="learning_rate")
    parser.add_argument("--data_root", type=str,
                        default='',
                        help="data_path")
    parser.add_argument("--csv_path", type=str,
                        default='',
                        help="csv_path")
    parser.add_argument("--log_path", type=str, default='log/tensorboard/0001/',
                        help="data_path")
    parser.add_argument("--save_name", type=str, default='transformer_convnext_640_640_crop_1024_ce_cancer_9',
                        help="save_name")
    parser.add_argument("--model", type=str, default='transformer',
                        help="fmc,fmc0,lstm,textcnn,transformer,att_lstm")
    parser.add_argument("--pretrain", type=str, default='',
                        help="")

    opt = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    EPOCH = opt.epoch
    learning_rate = opt.learning_rate
    pretrain_w_path = opt.pretrain_weight_path if 'pretrain_weight_path' in opt else ''

    writer = SummaryWriter(os.path.join(opt.log_path, opt.save_name), comment=opt.save_name, flush_secs=2)
    save_path = os.path.join('save', '20240109', opt.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(str(opt))

    train_data = FeatureMapDataset(opt.csv_path, opt.data_root, is_train=True, normalize=False,
                                        need_random=True, skip_header=True)
    test_data = FeatureMapDataset(opt.csv_path, opt.data_root, is_train=False, normalize=False, need_random=False,
                                       skip_header=True)

    # print('Using att_lstm')
    # from models.bilstm_attention import BiLSTM_Attention
    BATCH_SIZE = 8
    n_samples = BATCH_SIZE * 20
    # model = BiLSTM_Attention(num_class=len(train_data.classes_names), need_embedding=True, embedding_dim=1024,
    #                          hidden_dim=512)

    print('Using Transformer')
    from models.transformer import Transformer

    model = Transformer(num_class=len(train_data.classes_names))
    if opt.pretrain != '':
        state_dict = torch.load(opt.pretrain)
        model.load_state_dict(state_dict)

    sampler = WeightedRandomSampler(weights=train_data.sample_weight, num_samples=n_samples,
                                    replacement=True)
    train_data_loader = DataLoaderX(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True,
                                    sampler=sampler)
    test_data_loader = DataLoaderX(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                                   pin_memory=True) 
    optimizer = optim.RAdam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    # from models.losses.focal_loss import FocalLoss
    # criterion = FocalLoss(class_num=len(train_data.classes_names))
    criterion = nn.CrossEntropyLoss()

    # from models.losses.tree_loss import TreeLoss
    #
    # criterion = TreeLoss()

    trainer = Trainer(model, criterion=criterion, classes_names=train_data.classes_names, optimizer=optimizer,
                      writer=writer)
    max_f1 = 0
    max_auc = 0
    max_m_score = 0
    max_avg_recall = 0
    min_train_loss = float('inf')
    min_test_loss = float('inf')
    for epoch in range(1, EPOCH + 1):
        print('-------------------------------%d------------------------------------------' % (epoch))
        train_loss, train_acc = trainer.train(train_data_loader, n_samples, epoch)
        if epoch >= 1:
            val_loss, acc, cm, avg_f1, weighted_f1, auc,avg_recall,m_score = trainer.validate(
                test_data_loader, test_data.__len__(), epoch)
            to_save = True

            if avg_f1 >= max_f1:
                max_f1 = avg_f1
                to_save = True
            if auc > max_auc:
                max_auc = auc
                to_save = True
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                to_save = True
            if val_loss <= min_test_loss:
                min_test_loss = val_loss
                to_save = True
            if avg_recall > 0.75:
                max_avg_recall = avg_recall
                to_save = True
            if m_score > max_m_score:
                max_m_score = m_score
                to_save = True
            if to_save:
                temp_model_name = "ckpt_%d.pth" % (epoch)
                ckpt_name = os.path.join(save_path, temp_model_name)
                torch.save(trainer.model.module.state_dict(), ckpt_name)
