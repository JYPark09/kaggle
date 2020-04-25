import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

cv_params = {
        'max_depth': np.arange(1, 10, 1),
        'learning_rate': np.arange(0.05, 0.6, 0.05), 
        'n_estimators': np.arange(50, 300, 50)
}

fix_params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 2
}

dataset = np.load('data/train.npy')

features = dataset[:, 1:7]
target = dataset[:, 7]

csv = GridSearchCV(XGBClassifier(**fix_params),
                cv_params,
                scoring='accuracy',
                cv=4,
                n_jobs=8)
csv.fit(features, target)

print('Best score: {}'.format(csv.best_score_))
print(csv.best_params_)

test = np.load('data/test.npy')

features_test = test[:, 1:8]
pred = csv.predict(features_test)

submission = pd.concat([pd.Series(test[:, 0].astype(int)), pd.Series(pred.astype(int))], axis='columns')
submission.columns = ['PassengerId', 'Survived']

submission.to_csv('submission.csv', header=True, index=False)

