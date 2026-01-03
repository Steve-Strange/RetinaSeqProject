import os
import random
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8).reshape(-1)

    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    # labels=[0, 1] 防止缺失某一类报错
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    score_f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    score_jaccard = tp / (tp + fp + fn + 1e-6)
    score_recall = tp / (tp + fn + 1e-6)
    score_specificity = tn / (tn + fp + 1e-6)
    score_precision = tp / (tp + fp + 1e-6)
    score_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_specificity]