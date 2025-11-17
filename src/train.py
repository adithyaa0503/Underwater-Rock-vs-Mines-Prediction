import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


from preprocess import load_data, split_and_scale
from utils import ensure_dirs, save_model, save_scaler, plot_and_save_confusion_matrix




def evaluate_and_print(name, model, X_test, y_test, labels):
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label='M')
rec = recall_score(y_test, y_pred, pos_label='M')
print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
plot_and_save_confusion_matrix(y_test, y_pred, labels, name)
return acc, prec, rec




def main():
ensure_dirs()


X, y = load_data('data/sonar_data.csv')
X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
save_scaler(scaler)


labels = ['R', 'M']


# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
save_model(lr, 'logistic_regression')
lr_metrics = evaluate_and_print('Logistic Regression', lr, X_test, y_test, labels)


# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
save_model(knn, 'knn_k5')
knn_metrics = evaluate_and_print('KNN (k=5)', knn, X_test, y_test, labels)


# SVM (RBF)
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train, y_train)
save_model(svm, 'svm_rbf')
svm_metrics = evaluate_and_print('SVM (RBF)', svm, X_test, y_test, labels)


print('\nAll models trained and saved in the models/ folder.')




if __name__ == '__main__':
main()
