from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np


def metrics(y_true, y_pred):
    all_metrics = {}

    all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)

    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    all_metrics['acc'] = accuracy_score(y_true, y_pred)
    return all_metrics