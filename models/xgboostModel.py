
from preprocessing.preproc import dataframe
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV
from helper_functions.evaluate import evaluate
param_dist = {
    'n_estimators': randint(2, 100),
    'max_depth': randint(2, 10),
    'learning_rate': uniform(0.01, 0.29)
}

#pretend we have x_test,y_test, x_train, y_train
xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
xgb = RandomizedSearchCV(xgb, param_dist, cv=5)
xgb.fit(x_train, y_train)
prediction = xgb.best_estimator_.predict(x_test)
prediction_prob = xgb.best_estimator_.predict_proba(x_test)[:, 1]
# Use cross-validation to get more reliable performance scores
scores = cross_val_score(xgb.best_estimator_, x_train, y_train, cv=5, scoring='accuracy')

print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")

evaluate(y_test,prediction, prediction_prob)



