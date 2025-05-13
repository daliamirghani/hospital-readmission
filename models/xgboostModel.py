from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from helper_functions.evaluate import evaluate
from preprocessing.preproc import X_test, Y_test, X_train, Y_train
from numpy import sum

# Handle class imbalance
neg, pos = sum(Y_train == 0), sum(Y_train == 1)
print("neg", neg, "pos", [pos])
scale = neg / pos

# Define parameter space
# param_dist = {
#     'n_estimators': randint(50, 200),
#     'max_depth': randint(3, 15),
#     'learning_rate': uniform(0.01, 0.5),
#     'subsample': uniform(0.6, 0.4),  # stays in [0.6, 1.0]
#     'colsample_bytree': uniform(0.6, 0.4),
#     'min_child_weight': randint(1, 10),
#     'gamma': uniform(0, 0.5),
#     'reg_alpha': uniform(1, 10),
#     'reg_lambda': uniform(10, 100)
# }


base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')
rfe = RFE(estimator=base_model, n_features_to_select=40)
rfe.fit(X_train, Y_train)
X_train_sel = X_train.loc[:, rfe.support_]
X_test_sel = X_test.loc[:, rfe.support_]
selected_features = X_train.columns[rfe.support_]
print("Selected features:", selected_features)


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic', scale_pos_weight=scale)
# Hyperparameter tuning
# search = RandomizedSearchCV(xgb, param_distributions=param_dist, cv=5, scoring='f1', n_iter=20)
# search.fit(X_train_sel, Y_train)
xgb.fit(X_train_sel, Y_train)
# Evaluation
# best_model = search.best_estimator_
predictions = xgb.predict(X_test_sel)
predictions_prob = xgb.predict_proba(X_test_sel)[:, 1]
# threshold = 0.28
# predictions = (predictions_prob > threshold).astype(int)

scores = cross_val_score(xgb, X_train_sel, Y_train, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean()}")

evaluate(Y_test, predictions, predictions_prob)

