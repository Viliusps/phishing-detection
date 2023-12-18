import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, DetCurveDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff
import joblib
import numpy as np
import tensorflow as tf


def plotROCCurve_multi(models, X_test, y_test):
    plt.figure(figsize=(10, 6))

    for model in models:
        y_score = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)

        plt.plot(fpr, tpr, lw=2,
                 label=f'{model.__class__.__name__} (AUC = {auc_roc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plotPrecisionRecallCurve_multi(models, X_test, y_test):
    plt.figure(figsize=(10, 6))

    for model in models:
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = skplt.metrics.precision_recall_curve(
            y_test, y_prob)
        plt.plot(recall, precision, lw=2, label=f'{model.__class__.__name__}')

    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.show()


def plotDetCurve_multi(models, X_test, y_test):
    fig, ax_det = plt.subplots(figsize=(8, 8))

    for model in models:
        det_curve_display = DetCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax_det, name=str(model))
        model_name = model.__class__.__name__
        det_curve_display.line_.set_label(model_name)

    ax_det.set_xlabel('False Positive Rate')
    ax_det.set_ylabel('Miss Rate')
    ax_det.set_title('Detection Error Tradeoff Curve')
    ax_det.grid(linestyle='--')
    ax_det.legend()

    plt.show()


def create_model(optimizer='adam', activation='tanh', neurons_layer_one=30, hidden_neurons=12, dropout_rate=0):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(neurons_layer_one,
              input_shape=(30,), activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(hidden_neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    knn_model = joblib.load('models/knn_model.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    nn_model = joblib.load('models/nn_model.pkl')
    vclf_model = joblib.load('models/vclf_model.pkl')

    models = [knn_model, rf_model, xgb_model, svm_model, nn_model, vclf_model]

    data, meta = arff.loadarff('dataset.arff')
    df = pd.DataFrame(data)
    for column in df.columns:
        df[column] = df[column].str.decode('utf-8').astype(int)
    X = df.drop('Result', axis=1)
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    le = LabelEncoder()
    y_test = le.fit_transform(y_test)
    # plotROCCurve_multi(models, X_test, y_test)
    # plotPrecisionRecallCurve_multi(models, X_test, y_test)
    plotDetCurve_multi(models, X_test, y_test)
