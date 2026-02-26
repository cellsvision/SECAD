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
from einops import rearrange, reduce, asnumpy, parse_shape
from scipy import ndimage
from torch.cuda.amp import autocast as autocast, GradScaler
from utils.parallel import DataParallelModel, DataParallelCriterion

class Trainer(object):
    def __init__(self, model, classes_names, criterion=None, optimizer=None, train_task='classification', writer=None):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device( "cpu")
        print("device:", self.device)

        # model = DataParallelModel(model)
        model = nn.DataParallel(model)

        self.model = model.to(self.device)
        # self.criterion = DataParallelCriterion(criterion)
        self.criterion = criterion

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
        # self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
        # self.criterion = FocalLoss(class_num=num_classes,alpha=torch.tensor([[0.25],[1]])).to(device)        # self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
        # if criterion==None:
        #     criterion=nn.CrossEntropyLoss()
        # self.criterion = criterion.to(self.device)
        self.classes_names = classes_names
        print("---optimizer:", self.optimizer)
        print("---criterion:", self.criterion)
        self.writer = writer
        self.train_task = train_task
        self.scaler = GradScaler()

    def train(self, train_loader, n_data, epoch):
        self.model.train()
        train_loss = 0.0
        correct_count = torch.tensor(0, dtype=torch.int16).to(self.device)
        start = time.time()
        for i, training_data in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()

            imgs, labels, id, hospital = training_data
            imgs = imgs.to(self.device)
            # gender = gender.to(self.device)
            # age = age.to(self.device)

            labels = labels.to(self.device)
            with autocast():
                if self.criterion is not None:
                    # output = self.model(imgs)
                    output = self.model(imgs)
                    if self.train_task == 'regression':
                        output = output.squeeze(1)
                        labels = labels.float()
                        # output = self.model(imgs)
                    elif self.train_task == 'segmentation':
                        n, _, d, h, w = output.shape
                        new_label_masks = np.zeros([n, d, h, w])
                        for label_id in range(n):
                            label_mask = labels[label_id]
                            [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                            label_mask = torch.reshape(label_mask, [ori_d, ori_h, ori_w])
                            scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                            label_mask = ndimage.interpolation.zoom(label_mask.cpu().numpy(), scale, order=0)
                            new_label_masks[label_id] = label_mask
                        new_label_masks = torch.tensor(new_label_masks).to(torch.int64).to(self.device)
                        labels = new_label_masks
                    loss = self.criterion(output, labels)
                else:
                    loss, output = self.model(imgs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()  
            train_loss += loss.item()

            if self.train_task == 'classification':
                pred = torch.argmax(output.data, dim=1)
            elif self.train_task == 'regression':
                pred = torch.round(output).int()
            elif self.train_task == 'segmentation':
                pred = torch.argmax(output.data, dim=1)

            result = labels.int() == pred
            sum_result = torch.sum(result)
            correct_count = correct_count + sum_result
        # print('---[Epoch: %d]' % (epoch))
        acc = torch.true_divide(correct_count, n_data)
        end = time.time()
        if self.train_task != 'segmentation':

            print('---Training Loss: %.3f, Training Accuracy: %i/%i=%f,Time:%f' % (
                train_loss, correct_count.data, n_data, acc, end - start))
        else:
            print(
                '+++Testing Loss: %.3f, Training Correct Count: %i' % (train_loss, correct_count.data))
        if self.writer is not None:
            self.writer.add_scalar('Train Loss', train_loss, global_step=epoch)
        return train_loss, acc

    def validate(self, val_loader, n_data, epoch):

        self.model.eval()

        with torch.no_grad():
            val_loss = 0.0
            correct_count = 0
            id_list = []
            all_labels = []
            all_predictions = []
            all_outputs = []
            hospital_dict = {}
            for i, val_data in enumerate(val_loader, 0):
                imgs, labels, id, hospitals = val_data
                imgs = imgs.to(self.device)
                # gender = gender.to(self.device)
                # age = age.to(self.device)
                labels = labels.to(self.device)
                id_list += list(id)
                hospitals = np.array(hospitals)
                if self.criterion is not None:

                    with autocast():
                        # output = self.model(imgs)
                        output = self.model(imgs)
                    if self.train_task == 'regression':
                        output = output.squeeze(1)
                        labels = labels.float()
                    elif self.train_task == 'segmentation':
                        n, _, d, h, w = output.shape
                        new_label_masks = np.zeros([n, d, h, w])
                        for label_id in range(n):
                            label_mask = labels[label_id]
                            [ori_c, ori_d, ori_h, ori_w] = label_mask.shape
                            label_mask = torch.reshape(label_mask, [ori_d, ori_h, ori_w])
                            scale = [d * 1.0 / ori_d, h * 1.0 / ori_h, w * 1.0 / ori_w]
                            label_mask = ndimage.interpolation.zoom(label_mask.cpu().numpy(), scale, order=0)
                            new_label_masks[label_id] = label_mask
                        new_label_masks = torch.tensor(new_label_masks).to(torch.int64).to(self.device)
                        labels = new_label_masks
                    # output = self.model(imgs)
                    loss = self.criterion(output, labels)
                else:
                    loss, output = self.model(imgs, labels)
                val_loss += loss.item()
                if self.train_task == 'classification':
                    output = torch.softmax(output.data, dim=1)
                    pred = torch.argmax(output, dim=1)
                elif self.train_task == 'regression':
                    pred = torch.round(output.data).int()
                    floor = torch.floor(output.data, )
                    ceil = torch.ceil(output.data)
                    gap = output - floor
                    gap = rearrange(gap, "s -> s 1")
                    floor = rearrange(floor, "s -> s 1")
                    ceil = rearrange(ceil, "s -> s 1")
                    one_hot = torch.zeros((output.size(0), len(self.classes_names)), device=output.device)
                    one_hot = one_hot.scatter_(1, floor.data.to(torch.int64), gap)
                    one_hot = one_hot.scatter_(1, ceil.data.to(torch.int64), 1 - gap)
                    output = one_hot
                elif self.train_task == 'segmentation':
                    pred = torch.argmax(output.data, dim=1)
                else:
                    raise Exception('No Target Task!!!')
                result = labels.int() == pred
                correct_count = correct_count + torch.sum(result).int()
                all_labels.append(labels)
                all_predictions.append(pred)
                all_outputs.append(output)

                labels = labels.cpu().float().numpy()
                pred = pred.cpu().float().numpy()
                output = output.cpu().float().numpy()
                for hospital_name in val_loader.dataset.hospital_set:
                    target_hosp_idx = hospitals == hospital_name

                    hosp_labels, hosp_predictions, hosp_outputs = hospital_dict.get(hospital_name, [[], [], []])
                    hosp_labels.append(labels[target_hosp_idx])
                    hosp_predictions.append(pred[target_hosp_idx])
                    hosp_outputs.append(output[target_hosp_idx])
                    hospital_dict[hospital_name] = [hosp_labels, hosp_predictions, hosp_outputs]
            all_labels = torch.cat(all_labels, dim=0)
            all_predictions = torch.cat(all_predictions, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)

            all_outputs = torch.softmax(all_outputs.float().cpu(), dim=1)


            n_data = len(val_loader.sampler)
            acc = torch.true_divide(correct_count, n_data)

            print(
                '+++Val Loss: %.3f, Val Accuracy: %i/%i=%f' % (val_loss, correct_count.data, n_data, acc))
            # sensitivity_list, specificity_list, cmt = draw_confusion_matrix(all_labels, all_predictions,
            #                                                                 self.classes_names)
            cm,  avg_f1, weighted_f1,m_score = plot_testing_results(
                all_labels.cpu().numpy(), all_predictions.cpu().numpy(), [i for i in range(len(self.classes_names))],
                self.classes_names)
            # avg_precision, avg_recall, avg_f1, weighted_f1 = get_classification_report(all_labels, all_predictions,
            #                                                                            self.classes_names)
            avg_recall = recall_score(all_labels.cpu().numpy(), all_predictions.cpu().numpy(), average='macro')
            auc = get_auc(all_labels, all_outputs)
            for hospital_name in val_loader.dataset.hospital_set:
                print('-------------%s-------------' % hospital_name)
                hosp_labels, hosp_predictions, hosp_outputs = hospital_dict.get(hospital_name, [[], [], []])
                all_labels = np.concatenate(hosp_labels, axis=0)
                all_predictions = np.concatenate(hosp_predictions, axis=0)
                all_outputs = np.concatenate(hosp_outputs, axis=0)
                if hospital_name!='test':
                    plot_testing_results(all_labels, all_predictions, [i for i in range(len(self.classes_names))],
                                         self.classes_names)
                    get_auc(all_labels, all_outputs)
                else:
                    cm,  avg_f1, weighted_f1,m_score = plot_testing_results(all_labels, all_predictions, [i for i in range(len(self.classes_names))],
                                         self.classes_names)
                    avg_recall = recall_score(all_labels, all_predictions, average='macro')
                    auc = get_auc(all_labels, all_outputs)
            if self.writer is not None:
                self.writer.add_scalar('Testing Acc', acc.item(), global_step=epoch)
                self.writer.add_scalar('Test Loss', val_loss, global_step=epoch)
                self.writer.add_scalar('Avg F1', avg_f1, global_step=epoch)
                self.writer.add_scalar('AUC', auc, global_step=epoch)
                self.writer.add_scalar('Weighted F1', weighted_f1, global_step=epoch)
            return val_loss, acc, cm, avg_f1, weighted_f1, auc,avg_recall,m_score
