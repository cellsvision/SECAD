import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torch
# torch.cuda.empty_cache()

import argparse
import torch_optimizer as optim
from tensorboardX import SummaryWriter
from utils.trainer2 import Trainer
import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
import warnings
from utils.model_util import get_model

# from torch.cuda.amp import GradScaler
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    # python -m visdom.server
    # mp.set_start_method('spawn')

    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()

    # parser.add_argument("--pretrained_weights", type=str, default="", help="if specified starts from checkpoint model")

    parser.add_argument("--epoch", type=int, default=250, help="epoch")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning_rate")
    parser.add_argument("--image_size", type=tuple, default=(640, 640), help="image_size")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--batch_num", type=int, default=1000, help="batch_num")
    parser.add_argument("--xml_root", type=str,
                        default='', help="data_root")
    parser.add_argument("--train_data", type=str,
                        default='', help="data_root")
    parser.add_argument("--val_data", type=str,
                        default='', help="val_data")
    
    parser.add_argument("--log_path", type=str, default='log/tensorboard/0001/',
                        help="data_path")
    parser.add_argument("--save_name", type=str, default='_ce_out_box_adc_micro1',
                        help="save_name")
    parser.add_argument("--model_name", type=str, default='convnext',
                        help="cbam18,cbam34,cbam50,gcn_s,lstm,efficientnet_b3,convnext,convnext_large,transfg,moganet_base,convnextv2_base")

    opt = parser.parse_args()
    EPOCH = opt.epoch
    BATCH_SIZE = opt.batch_size
    learning_rate = opt.learning_rate
    pretrain_w_path = opt.pretrain_weight_path if 'pretrain_weight_path' in opt else ''
    opt.save_name = opt.model_name + '_%i_%i' % opt.image_size + opt.save_name

    writer = SummaryWriter(os.path.join(opt.log_path, opt.save_name), comment=opt.save_name, flush_secs=2)
    save_path = os.path.join('save', '00001', opt.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(str(opt))

    from datasets.PatchDataset import PatchDataset

    train_data = PatchDataset(opt.train_data, xml_root=opt.xml_root, img_size=opt.image_size, is_train=True)
    val_data = PatchDataset(opt.val_data, xml_root=opt.xml_root, img_size=opt.image_size, is_train=False)

    n_samples = BATCH_SIZE * opt.batch_num
    sampler = WeightedRandomSampler(weights=train_data.sample_weight, num_samples=n_samples, replacement=True)

    from utils.DataLoaderX import DataLoaderX

    train_data_loader = DataLoaderX(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=24, pin_memory=True,
                                    sampler=sampler)
    # train_data_loader = DataLoaderX(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    val_data_loader = DataLoaderX(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                  pin_memory=True)  

    model = get_model(opt.model_name, need_head=True, pretrained=True, num_classes=len(train_data.classes_names), )

    # state_dict = torch.load(')

    # model.load_state_dict(state_dict)

    optimizer = optim.RAdam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    if '_ce_' in opt.save_name:
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        train_task = 'softmax'
    elif '_mclw_' in opt.save_name:
        from models.losses.multi_cls_weight_loss import Multi_Class_Weight_Loss

        matrix = [
            [0, 1, 3],
            [2, 0, 0.5],
            [3, 1, 0],
        ]
        criterion = Multi_Class_Weight_Loss(class_num=len(train_data.classes_names), alpha_matrix=matrix)
        train_task = 'softmax'
    elif '_rank_' in opt.save_name:
        from models.losses.ranking_loss import RankingLoss

        criterion = RankingLoss()
        train_task = 'rank'
    else:
        # from models.losses.focus_loss import Focus_loss
        # from models.losses.level_loss import Focus_loss
        from models.losses.level_loss2 import Focus_loss

        criterion = Focus_loss()
        train_task = 'descend'
    print('---Train Task', train_task)
    trainer = Trainer(model, criterion=criterion, classes_names=train_data.classes_names, optimizer=optimizer,
                      writer=writer, train_task=train_task)
    max_f1 = 0
    max_auc = 0
    max_acc = 0
    max_weighted_f1 = 0
    max_idvd_acc = 0
    max_whole_acc = 0
    min_train_loss = float('inf')
    min_val_loss = float('inf')

    for epoch in range(1, EPOCH + 1):
        print('-------------------------------%d------------------------------------------' % (epoch))
        train_loss = trainer.train(train_data_loader, epoch)
        if epoch >= 1:

            val_loss, acc, avg_precision, avg_recall, avg_f1, weighted_f1, auc = trainer.validate(
                val_data_loader, epoch)
            save_flag = False

            if auc > max_auc:
                max_auc = auc
                save_flag = True
            if train_loss <= min_train_loss:
                min_train_loss = train_loss
                save_flag = True
            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                save_flag = True
            if acc >= max_acc:
                max_acc = acc
                save_flag = True
            if avg_f1 >= max_f1 or avg_f1>0.8:
                max_f1 = avg_f1
                save_flag = True
            if weighted_f1 >= max_weighted_f1:
                max_weighted_f1 = weighted_f1
                save_flag = True
            if save_flag:
                temp_model_name = "%s_ckpt_%d.pth" % (opt.save_name, epoch)
                ckpt_name = os.path.join(save_path, temp_model_name)
                torch.save(trainer.model.module.state_dict(), ckpt_name)
