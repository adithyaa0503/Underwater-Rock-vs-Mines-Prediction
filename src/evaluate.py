import joblib
from preprocess import load_data, split_and_scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from utils import plot_and_save_confusion_matrix




def load_models():
lr = joblib.load('models/logistic_regression.pkl')
knn = joblib.load('models/knn_k5.pkl')
svm = joblib.load('models/svm_rbf.pkl')
scaler = joblib.load('models/scaler.pkl')
return lr, knn, svm, scaler




def main():
X, y = load_data('data/sonar_data.csv')
X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)


lr, knn, svm, scaler = load_models()


models = [('Logistic Regression', lr), ('KNN (k=5)', knn), ('SVM (RBF)', svm)]
labels = ['R', 'M']


for name, model in models:
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='M')
rec = recall_score(y_test, y_pred, pos_label='M')
print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
plot_and_save_confusion_matrix(y_test, y_pred, labels, name)




if __name__ == '__main__':
main()
