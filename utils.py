import math
import numpy as np
import pandas as pd


def sigmoid(z):
    return  1 / (1 + np.exp(-z))

def cross_entropy_loss(pred, true):
    loss = -1 * np.mean(true * np.log(pred) + (1 - true) * np.log(1 - pred))
    return loss

def cost(pred, true):
    loss = cross_entropy_loss(pred, true)
    m = pred.shape[0]
    final_cost = (1 / m) * np.sum(loss)
    return final_cost


# data processing

def feature_conv():
    df = pd.read_csv("stroke.csv")
    X = df[["age","avg_glucose_level","bmi"]].values
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    y = df["stroke"].map({"Stroke": 1, "No Stroke": 0}).values
    return X, y
