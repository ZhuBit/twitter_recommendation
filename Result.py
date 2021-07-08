import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, log_loss


class Result:
    def __init__(self, model_path='none', model='none', params='none', accuracy=0, precision=0, recall=0, f1=0,
                 specificity=0, log_loss=0):
        self.model_path = model_path
        self.model = model
        self.params = params
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.specificity = specificity
        self.log_loss = log_loss

    def __str__(self):
        return "Model: {0}, Params: {1}, Accuracy: {2}, Precision: {3}, Recall: {4}, F1: {5}, Specificity: {6}," \
               " Log_loss: {7}" \
            .format(self.model, self.params, self.accuracy, self.precision, self.recall, self.f1, self.specificity,
                    self.log_loss)

    def to_tuple(self):
        return (
        self.model_path, self.model, self.params, self.accuracy, self.precision, self.recall, self.f1, self.specificity,
        self.log_loss)

    def from_dataframe(self, df):
        return self.__init__(df['Model Path'].values[0], df['Model'].values[0], df['Params'], df['Accuracy'].values[0],
                     df['Precision'].values[0], df['Recall'].values[0], df['F1'].values[0], df['Specificity'].values[0],
                     df['Log Loss'].values[0])

    def store_result(self):
        with open('results/results.csv', 'a') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(self.to_tuple())

        print('result saved')

    def calculate_and_store_metrics(self, y_true, y_pred):
        self.accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        self.precision = precision_score(y_true=y_true, y_pred=y_pred, pos_label=1, zero_division=1)
        self.recall = recall_score(y_true=y_true, y_pred=y_pred, pos_label=1)
        self.specificity = recall_score(y_true=y_true, y_pred=y_pred, pos_label=0)
        self.f1 = f1_score(y_true=y_true, y_pred=y_pred, pos_label=1)
        self.log_loss = log_loss(y_true=y_true, y_pred=y_pred)


