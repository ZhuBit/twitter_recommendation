from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, log_loss


def generate_scores(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        "precision": precision_score(y_true=y_true, y_pred=y_pred, pos_label=1),
        "recall": recall_score(y_true=y_true, y_pred=y_pred, pos_label=1),  # sensitivity
        "specificity": recall_score(y_true=y_true, y_pred=y_pred, pos_label=0),  # specificity is recall with 0 as pos
        "f1": f1_score(y_true=y_true, y_pred=y_pred, pos_label=1),
        "log_loss": log_loss(y_true=y_true, y_pred=y_pred),
        # "roc_auc": roc_auc_score(y_true=y_true, y_score=y_score),
    }
