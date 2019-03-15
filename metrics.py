from sklearn.metrics import roc_auc_score


def mean(item):
    return sum(item) / len(item)


def cal_acc(true_y, pred_y):
    true_y = [label[0] for label in true_y]
    pred_y = [label[0] for label in pred_y.tolist()]

    count = 0
    for i in range(len(true_y)):
        if true_y[i] == pred_y[i]:
            count += 1
    acc = round((count / len(true_y)), 3)

    return acc


def cal_auc(true_y, pred_y):
    true_y = [label[0] for label in true_y]
    pred_y = [label[0] for label in pred_y.tolist()]
    return roc_auc_score(true_y, pred_y)


def cal_precision(true_y, pred_y):
    true_y = [label[0] for label in true_y]
    pred_y = [label[0] for label in pred_y.tolist()]

    tp = 0
    pred_pos = 0
    for i in range(len(true_y)):
        if pred_y[i] == 1:
            pred_pos += 1
            if pred_y[i] == true_y[i]:
                tp += 1

    precision = tp / pred_pos if pred_pos > 0 else 0

    return precision


def cal_recall(true_y, pred_y):
    true_y = [label[0] for label in true_y]
    pred_y = [label[0] for label in pred_y.tolist()]

    tp = 0
    true_pos = 0
    for i in range(len(true_y)):
        if true_y[i] == 1:
            true_pos += 1
            if pred_y[i] == true_y[i]:
                tp += 1
    recall = tp / true_pos if true_pos > 0 else 0

    return recall


def cal_f1(true_y, pred_y):
    precision = cal_precision(true_y, pred_y)
    recall = cal_recall(true_y, pred_y)

    if precision == 0 and recall == 0:
        return 0

    f1 = round((2 * precision * recall / (precision + recall)), 3)

    return f1

