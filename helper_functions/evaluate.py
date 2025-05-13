from sklearn.metrics import precision_score,recall_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score
def evaluate(Y_test, Y_pred,Y_pred_proba):
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Precision:", precision_score(Y_test, Y_pred,zero_division=0))
    print('Recall:', recall_score(Y_test, Y_pred))
    print('F1 score:', f1_score(Y_test, Y_pred))
    print("Confusion Matrix:", confusion_matrix(Y_test, Y_pred))
    print("ROC AUC:",roc_auc_score(Y_test, Y_pred_proba))



