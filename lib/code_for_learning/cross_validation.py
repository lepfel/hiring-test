# coding: utf-8
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from features import FeatureMaker

def train_model(train, list_of_features, model, target_name='is_bad'):
    """
    тренируем модель на train выборке
    """
    model.fit(train[list_of_features], train[target_name])
    return model


def predict_with_model(test, list_of_features, model):
    """
    записываем результат в виде index/pred как требуется в задании
    """
    prediction = model.predict_proba(test[list_of_features])[:, 1]
    target_prediction = pd.DataFrame()
    target_prediction['index'] = range(test.shape[0])
    target_prediction['prediction'] = prediction
    return target_prediction


def score_test(test, test_target_prediction):
    """
    смотрим roc-auc по категориям на собственном отложенном тесте
    """
    score_df = test[['category', 'is_bad']]
    score_df.loc[:, 'prediction'] = test_target_prediction.loc[:, 'prediction']
    category_aucs = score_df.groupby(['category']).apply(lambda x: roc_auc_score(y_true=x['is_bad'],
                                                                                 y_score=x['prediction']
                                                                                 ))
    print(category_aucs)
    return category_aucs.mean()


def cross_validation(train, list_of_features, model):
    """
    кросс-валидация, смотрим roc-auc по категориям
    """
    aucs_list = []
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(train):
        X_train, X_test = train.loc[train_index, list_of_features], train.loc[test_index, list_of_features]
        y_train, y_test = train.loc[train_index, 'is_bad'], train.loc[test_index, 'is_bad']
        category_test = train.loc[test_index, ['category', 'is_bad']]

        model.fit(X_train, y_train)
        kfold_prediction = model.predict_proba(X_test)[:, 1]

        category_test['prediction'] = kfold_prediction
        aucs_by_category = category_test.groupby(['category']).apply(lambda x: roc_auc_score(y_true=x['is_bad'],
                                                                                             y_score=x['prediction']
                                                                                             ))
        print(aucs_by_category)

        mean_aucs_by_category = aucs_by_category.mean()
        print(mean_aucs_by_category)
        aucs_list.append(mean_aucs_by_category)
    print(aucs_list)
    print(np.mean(aucs_list))
    return np.mean(aucs_list)

# testing and cross-validating
# python lib/code_for_learning/cross_validation.py
if __name__ == "__main__":
    feature_maker = FeatureMaker()
    
    train = pd.read_csv("/hiring-test-data/train_part.csv")
    test = pd.read_csv("/hiring-test-data/test.csv")

   
    feature_maker.fit_and_transform_train(train)

    print(feature_maker.list_of_features)

    feature_maker.transform_test(test)
    model = LGBMClassifier(n_estimators=500, max_depth=4)

    aucs_mean = cross_validation(train, feature_maker.list_of_features,model )
    
    print("скор на кросс-валидации: {}".format(str(aucs_mean)))

    # скор на отложенном тесте
    # тренировка на всем трейне
    model = train_model(train, feature_maker.list_of_features, model)

    # предсказания на отложенном тесте
    target_prediction = predict_with_model(test, feature_maker.list_of_features, model)
    test_score = score_test(test, target_prediction)
    print("скор на отложенном тесте: {}".format(str(test_score)))