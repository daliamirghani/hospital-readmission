from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
def evaluate(y_test, y_pred,y_pred_proba):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print('Recall:', recall_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    print("ROC AUC:",roc_auc_score(y_test, y_pred_proba))



