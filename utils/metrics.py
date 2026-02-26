import numpy as np
import torch
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score


def confusion_matrix_2_numpy(y_true, y_pred, N=None):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    y_true = y_true.detach().cpu().to(torch.int16).numpy()
    y_pred = y_pred.detach().cpu().to(torch.int16).numpy()
    if (N is None):
        N = max(max(y_true), max(y_pred)) + 1
    y = N * y_true + y_pred
    y = np.bincount(y, minlength=N*N)
    y = y.reshape(N, N)
    return y

def get_classification_report(truths, preds,multi_flag=False):
    truths = truths.detach().cpu().to(torch.int16).numpy()
    preds = preds.detach().cpu().to(torch.int16).numpy()
    if not multi_flag:
        cmt = confusion_matrix(truths, preds)
    else:
        cmt = multilabel_confusion_matrix(truths, preds)
    print(cmt)
    report = classification_report(truths, preds)
    print(report)
    avg_precision = precision_score(truths, preds, average='macro')
    avg_recall = recall_score(truths, preds, average='macro')
    avg_f1 = f1_score(truths, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(truths, preds, average='weighted', zero_division=0)
    return avg_precision, avg_recall, avg_f1, weighted_f1


def get_auc(truths, outputs):
    try:
        if not isinstance(truths,np.ndarray):
            truths = truths.cpu().to(torch.int16).numpy()
            outputs = outputs.cpu().numpy()
        auc = roc_auc_score(truths, outputs, multi_class='ovr')
    except Exception as e:
        auc = 0
        print(e)
    print('---AUC:', auc)
    return auc


def get_m_score(spec, recall_lsil, recall_asc):
    # M_score2
    m_score = 1 / ((1 / sum([0.85, 0.95, 0.9])) * (1 / spec * 0.85 + 1 / recall_lsil * 0.95 + 1 / recall_asc * 0.9))
    # m_score = sum([spec * 0.85, recall_lsil * 0.95, recall_asc * 0.9]) / 2.7
    return m_score


def plot_testing_results(truth, prediction, labels, target_names):
    print(classification_report(truth, prediction, labels=labels, target_names=target_names))

    cm = confusion_matrix(truth, prediction, labels=labels)
    print(cm)

    print("-----Confusion Matrix----------\n")
    # if len(labels)<=3:
    #     confusion2 = np.array([[cm[0, 0], np.sum(cm[0, 1:])],
    #                            [np.sum(cm[1:, 0]), np.sum(cm[1:, 1:])]])
    #     #
    #     confusion3 = np.array([[cm[0, 0], cm[0, 1], np.sum(cm[0, 2:])],
    #                            [cm[1, 0], cm[1, 1], np.sum(cm[1, 2:])],
    #                            [np.sum(cm[2:, 0]), np.sum(cm[2:, 1]), np.sum(cm[2:, 2:])]])
    # else:
    #     confusion2 = np.array([[np.sum(cm[0:2, 0:2]), np.sum(cm[0:2, 2:])],
    #                            [np.sum(cm[2:, 0:2]), np.sum(cm[2:, 2:])]])
    #
    #     confusion3 = np.array([[np.sum(cm[0:2, 0:2]), np.sum(cm[0:2, 2]), np.sum(cm[0:2, 3:])],
    #                            [np.sum(cm[2, 0:2]), cm[2, 2], np.sum(cm[2, 3:])],
    #                            [np.sum(cm[3:, 0:2]), np.sum(cm[3:, 2]), np.sum(cm[3:, 3:])]])
    print('total samples:', sum(sum(cm)))
    # recall_asc_us = confusion2[1, 1] / np.sum(confusion2[1, :])
    # specificity_asc_us = confusion2[0, 0] / np.sum(confusion2[0, :])
    # recall_lsil = np.sum(confusion3[2, 1:]) / np.sum(confusion3[2, :])
    # specificity_lsil = confusion3[0, 0] / np.sum(confusion3[0, :])
    # specificity_nghuc = np.sum(cm[0, :2]) / np.sum(cm[0, :])
    # specificity_auc_n = np.sum(cm[1, :2]) / np.sum(cm[1, :])
    # print('Neg/Pos:\n', confusion2)
    # print('Recall for AUC+:%.4f' % (recall_asc_us))
    # print('Specificity: %.4f' % (specificity_asc_us))
    # print('\nNeg/Asc/Pos:\n', np.array(confusion3))
    # print('Recall for HGUC:%.4f' % (recall_lsil))
    # print('Specificity:%.4f' % (specificity_lsil))
    # print('specificity_nghuc:%.4f' % (specificity_nghuc))
    # print('specificity_auc_n:%.4f' % (specificity_auc_n))
    avg_f1 = f1_score(truth, prediction, average='macro', zero_division=0)
    weighted_f1 = f1_score(truth, prediction, average='weighted', zero_division=0)
    # m_score = get_m_score(specificity_asc_us, recall_lsil, recall_asc_us)
    # neg_precision = confusion2[0,0]/np.sum(confusion2[:,0])
    # pos_precision = confusion2[1,1]/np.sum(confusion2[:,1])
    # print('neg_precision',neg_precision)
    # print('pos_precision',pos_precision)
    spec = cm[0,0]/np.sum(cm[0])
    sens =cm[1,1]/np.sum(cm[1])
    m_score = 1 / ((1 / sum([1, 1])) * (
            1 / spec  + 1 / sens ))
    print('m_score:',m_score)
    return cm,  avg_f1, weighted_f1,m_score
