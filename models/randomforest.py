from helper_functions.evaluate import evaluate
from preprocessing.preproc import X_test, Y_test, X_train, Y_train
from sklearn.ensemble import RandomForestClassifier
# smote = SMOTE(random_state=42)
# X_train, Y_train = s mote.fit_resample(X_train, Y_train)
from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=42)
X_train, Y_train = smote_enn.fit_resample(X_train, Y_train)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
predictions_prob = classifier.predict_proba(X_test)[:, 1]
evaluate(Y_test, predictions, predictions_prob)