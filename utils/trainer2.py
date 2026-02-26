import numpy as np
import torch
import time
import torch.nn as nn
import torch_optimizer as optim
# from utils.grid import GridMask

from utils.metrics import *
from tqdm import tqdm

from models.mish import Mish
import torch.nn.functional as F
# from utils.util import draw_confusion_matrix
from einops import rearrange, reduce, asnumpy, parse_shape,repeat
from scipy import ndimage

from torch.cuda.amp import autocast as autocast,GradScaler

class Trainer(object):
    def __init__(self, model, classes_names, criterion=None, optimizer=None, train_task='classification',
                 writer=None, ):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", self.device)
        #
        model = nn.DataParallel(model)

        self.model = model.to(self.device)

        if optimizer == None:
            self.optimizer = optim.RAdam(
                self.model.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
            )
        else:
            self.optimizer = optimizer
        self.criterion = criterion
        self.classes_names = classes_names
        self.train_task = train_task
        self.writer = writer
        print("---optimizer:", self.optimizer)
        print("---criterion:", self.criterion)
        self.scaler = GradScaler()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

    def train(self, train_loader, epoch):
        self.model.train()
        train_loss = 0.0
        correct_count = 0
        start = time.time()
        for i, training_data in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()

            imgs, labels = training_data
            imgs = imgs.to(self.device)

            labels = labels.to(self.device)
            with autocast():
                if self.criterion is not None:
                    if self.train_task == 'rank':
                        output, _,_,_ = self.model(imgs)
                    else:
                        output,_ = self.model(imgs)

                    loss = self.criterion(output, labels)
                else:
                    loss, output = self.model(imgs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()  # 准备着，看是否要增大scaler
            train_loss += loss.item()
            if self.train_task == 'softmax':
                pred = torch.argmax(output.data, dim=1)
            elif self.train_task == 'descend':
                output = output.data.flip(1)
                output = torch.sigmoid(output)
                output = (output >= 0.5).to(torch.int8)
                pred = torch.argmax(output, dim=1)
                pred = len(self.classes_names) - pred - 1
            elif self.train_task == 'rank':
                pred = torch.argmax(output[:,:,1].data, dim=1)

            result = labels.int() == pred
            correct_count = correct_count + torch.sum(result).item()
        n_data = train_loader.sampler.num_samples
        acc = torch.true_divide(correct_count, n_data)
        end = time.time()

        print('---Training Loss: %.3f, Training Accuracy: %i/%i=%f,Time:%f' % (
            train_loss, correct_count, n_data, acc, end - start))

        if self.writer is not None:
            self.writer.add_scalar('Train Loss', train_loss, global_step=epoch)
        return train_loss

    def validate(self, val_loader, epoch):

        self.model.eval()

        with torch.no_grad():
            val_loss = 0.0
            correct_count = 0
            all_labels = []
            all_predictions = []
            all_outputs = []
            pos_img_count = 0
            neg_img_count = 0
            visual_pos_imgs=[]
            visual_neg_imgs=[]
            visual_pos_s_imgs=[]
            visual_neg_s_imgs=[]
            for i, valing_data in enumerate(val_loader, 0):
                imgs, labels = valing_data
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                with autocast():
                    if self.criterion is not None:
                        if self.train_task == 'rank':
                            output, _,s_imgs,atten1 = self.model(imgs)
                        else:
                            output,_ = self.model(imgs)[:2]

                        loss = self.criterion(output, labels)
                    else:
                        loss, output = self.model(imgs, labels)
                val_loss += loss.item()
                if self.train_task == 'softmax':
                    output = torch.softmax(output.data, dim=1)
                    pred = torch.argmax(output.data, dim=1)
                    all_outputs.append(output)
                elif self.train_task == 'descend':
                    output = output.data.flip(1)
                    output = torch.sigmoid(output)
                    pred = (output >= 0.5).to(torch.int8)
                    pred = torch.argmax(pred, dim=1, keepdim=False)
                    pred = len(self.classes_names) - pred - 1
                    output = output.data.flip(1)
                    for j, label in enumerate(labels):
                        all_outputs.append(output[j, label:])
                elif self.train_task == 'rank':
                    output = torch.softmax(output[:, :, 1].data, dim=1)
                    pred = torch.argmax(output.data, dim=1)
                    all_outputs.append(output)
                    if pos_img_count<5:
                        for j,label, in enumerate(labels):
                            if label>=1:
                                img = imgs[j]
                                img = img * repeat(self.std,'c -> c h w',h=img.size(1),w=img.size(2))
                                img = img + repeat(self.mean,'c -> c h w',h=img.size(1),w=img.size(2))
                                img = torch.round(img*255,decimals=0).int()
                                s_img = s_imgs[j]
                                s_img = s_img * repeat(self.std,'c -> c h w',h=s_img.size(1),w=s_img.size(2))
                                s_img = s_img + repeat(self.mean,'c -> c h w',h=s_img.size(1),w=s_img.size(2))
                                s_img = torch.round(s_img*255,decimals=0).int()
                                visual_pos_imgs.append(img.cpu())
                                visual_pos_s_imgs.append(s_img.cpu())
                                # pos_atten1_x.append(atten1[j][0])
                                # pos_atten1_y.append(atten1[j][1])
                                # pos_atten1_l.append(atten1[j][2])
                                pos_img_count+=1
                                if pos_img_count==5:
                                    break
                    if neg_img_count<5:
                        for j,label, in enumerate(labels):
                            if label==0:
                                img = imgs[j]
                                img = img * repeat(self.std,'c -> c h w',h=img.size(1),w=img.size(2))
                                img = img + repeat(self.mean,'c -> c h w',h=img.size(1),w=img.size(2))
                                img = torch.round(img*255,decimals=0).int()
                                s_img = s_imgs[j]
                                s_img = s_img * repeat(self.std,'c -> c h w',h=s_img.size(1),w=s_img.size(2))
                                s_img = s_img + repeat(self.mean,'c -> c h w',h=s_img.size(1),w=s_img.size(2))
                                s_img = torch.round(s_img*255,decimals=0).int()
                                visual_neg_imgs.append(img.cpu())
                                visual_neg_s_imgs.append(s_img.cpu())
                                neg_img_count+=1
                                if neg_img_count==5:
                                    break

                result = labels.int() == pred
                correct_count = correct_count + torch.sum(result).item()
                all_labels.append(labels)
                all_predictions.append(pred)

            if self.train_task == 'rank':
                visual_pos_imgs=torch.stack(visual_pos_imgs,dim=0)
                visual_pos_s_imgs=torch.stack(visual_pos_s_imgs,dim=0)
                visual_neg_imgs=torch.stack(visual_neg_imgs,dim=0)
                visual_neg_s_imgs=torch.stack(visual_neg_s_imgs,dim=0)
                self.writer.add_image('Pos Image', visual_pos_imgs, epoch, dataformats='NCHW')
                self.writer.add_image('Pos Image_small', visual_pos_s_imgs, epoch, dataformats='NCHW')
                self.writer.add_image('Neg Image', visual_neg_imgs, epoch, dataformats='NCHW')
                self.writer.add_image('Neg Image_small', visual_neg_s_imgs, epoch, dataformats='NCHW')
            all_labels = torch.cat(all_labels, dim=0)
            all_predictions = torch.cat(all_predictions, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)




            n_data = len(val_loader.sampler)
            acc = torch.true_divide(correct_count, n_data)

            print(
                '+++valing Loss: %.3f, valing Accuracy: %i/%i=%f' % (val_loss, correct_count, n_data, acc))
            cm = confusion_matrix_2_numpy(all_labels, all_predictions, len(self.classes_names))
            print(cm)
            avg_precision, avg_recall, avg_f1, weighted_f1 = get_classification_report(all_labels, all_predictions)
            if self.train_task == 'descend':
                all_outputs = all_outputs.flatten()
                new_label_list = []
                for label in all_labels:
                    new_label = torch.zeros(len(self.classes_names), dtype=torch.uint8)
                    new_label[label] = 1
                    new_label_list.append(new_label[label:])
                all_labels = torch.cat(new_label_list)


            auc = get_auc(all_labels, all_outputs)
            if self.writer is not None:
                self.writer.add_scalar('Val Loss', val_loss, global_step=epoch)
                self.writer.add_scalar('Avg F1', avg_f1, global_step=epoch)
                self.writer.add_scalar('Avg Precision', avg_precision, global_step=epoch)
                self.writer.add_scalar('Avg  Recall', avg_recall, global_step=epoch)
                self.writer.add_scalar('Accuracy', acc, global_step=epoch)
                self.writer.add_scalar('AUC', auc, global_step=epoch)
                self.writer.add_scalar('Weighted F1', weighted_f1, global_step=epoch)

            return val_loss, acc, avg_precision, avg_recall, avg_f1, weighted_f1, auc
