import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, precision_recall_curve, \
    f1_score, DetCurveDisplay, log_loss
import matplotlib.pyplot as plt
import scikitplot as skplt


def printAccuracy(model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
    accuracy = accuracy_score(y_train, y_pred_train)
    print(f'Train accuracy: {accuracy * 100:.2f}%')


def printLoss(model, X_train, X_test, y_train, y_test):
    y_prob = model.predict_proba(X_test)
    test_log_loss = log_loss(y_test, y_prob)
    print(f'Test Log Loss: {test_log_loss}')

    y_prob = model.predict_proba(X_train)
    test_log_loss = log_loss(y_train, y_prob)
    print(f'Train Log Loss: {test_log_loss}')


def printAUCandPvalues(y_test, y_score):
    auc_roc = roc_auc_score(y_test, y_score)
    p_value = 2 * (1 - auc_roc)
    print("AUC-ROC:", auc_roc)
    print("P-value:", p_value)


def printAUCPRandF1Scores(y_test, y_pred, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    auc_pr = auc(recall, precision)
    print("AUC-PR:", auc_pr)

    f1 = f1_score(y_test, y_pred)
    print("F1 Score:", f1)


def plotROCCurve(y_test, y_score):
    auc_roc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(auc_roc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plotDetCurve(model, X_test, y_test):
    DetCurveDisplay.from_estimator(model, X_test, y_test)


def printConfMatrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)


def plotConfMatrix(y_test, y_pred):
    skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
    plt.show()


def plotPrecisionRecallCurve(y_test, y_prob):
    skplt.metrics.plot_precision_recall_curve(y_test, y_prob)
    plt.show()


def plotFeatureImportance(model, X):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Top 10 Feature Importances')
    plt.show()
