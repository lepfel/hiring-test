from sklearn.metrics import roc_auc_score
import pandas as pd


def score_test(test, target_prediction):
    """
    тестирование пайплайна на собственном тесте
    """
    score_df = test[['category', 'is_bad']]
    score_df.loc[:, 'prediction'] = target_prediction.loc[:, 'prediction']
    category_aucs = score_df.groupby(['category']).apply(lambda x: roc_auc_score(y_true=x['is_bad'],
                                                                                 y_score=x['prediction']
                                                                                 ))
    print(category_aucs)
    return category_aucs.mean()


# тестирование на отложенном тесте
# python lib/scoring_test.py
if __name__ == "__main__":
    test = pd.read_csv('/hiring-test-data/test.csv')
    target_prediction = pd.read_csv('/hiring-test-data/prediction.csv')
    score = score_test(test, target_prediction)
    print(" усредненный ROC-AUC по каждой категории объявлений: {}".format(str(score)))
