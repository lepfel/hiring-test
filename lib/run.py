# coding: utf-8
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from code_for_learning.features import FeatureMaker
import gc

# python lib/run.py
if __name__ == '__main__':
    # генерируем признаки
    feature_maker = FeatureMaker()
    train = pd.read_csv("/hiring-test-data/train.csv")
    feature_maker.fit_and_transform_train(train)

    model = LGBMClassifier(n_estimators=500, max_depth=4)
    model.fit(train[feature_maker.list_of_features], train["is_bad"])

    # удаляем треин чтобы не занимал лишнюю память
    del train
    gc.collect()

    #делаем предсказание на тесте
    test = pd.read_csv('/hiring-test-data/test.csv')
    feature_maker.transform_test(test)

    target_prediction = pd.DataFrame()
    target_prediction['index'] = range(test.shape[0])
    target_prediction['prediction'] = model.predict_proba(test[feature_maker.list_of_features])[:, 1]
    target_prediction.to_csv('/hiring-test-data/prediction.csv', index=False)
