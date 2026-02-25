import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_testing_results(truth, prediction, labels=None, target_names=None):

    # usage:
    # plot_testing_results(gt, model_pred, [0, 1, 2, 3, 4], ['nilm', 'asc_us', 'lsil', 'hsil', 'agc'])

    if labels is None:
        labels = [0, 1, 2, 3, 4]
    if target_names is None:
        target_names = ['nilm', 'asc_us', 'lsil', 'hsil', 'agc']

    print(classification_report(truth, prediction, labels=labels, target_names=target_names))
    cm = confusion_matrix(truth, prediction, labels=labels)
    print(cm)

    print("             Confusion Matrix:\n")

    confusion2 = np.array([[cm[0, 0], np.sum(cm[0, 1:])],
                           [np.sum(cm[1:, 0]), np.sum(cm[1:, 1:])]])

    confusion3 = np.array([[cm[0, 0], cm[0, 1], np.sum(cm[0, 2:])],
                           [cm[1, 0], cm[1, 1], np.sum(cm[1, 2:])],
                           [np.sum(cm[2:, 0]), np.sum(cm[2:, 1]), np.sum(cm[2:, 2:])]])

    print('total samples:', sum(sum(cm)))

    print('Neg/Pos:\n', confusion2)
    print('Recall for ASC-US and above:%.4f' % (confusion2[1, 1] / np.sum(confusion2[1, :])))
    print('Specificity: %.4f' % (confusion2[0, 0] / np.sum(confusion2[0, :])))
    print('\nNeg/Asc/Pos:\n', np.array(confusion3))
    print('Recall for LSIL and above:%.4f' % (np.sum(confusion3[2, 1:]) / np.sum(confusion3[2, :])))
    print('Specificity:%.4f' % (confusion3[0, 0] / np.sum(confusion3[0, :])))

    recall_asc = (confusion2[1, 1] / np.sum(confusion2[1, :]))
    recall_lsil = (np.sum(confusion3[2, 1:]) / np.sum(confusion3[2, :]))
    spec = (confusion3[0, 0] / np.sum(confusion3[0, :]))
    return recall_asc, recall_lsil, spec
