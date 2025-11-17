import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




def ensure_dirs():
os.makedirs('models', exist_ok=True)
os.makedirs('images', exist_ok=True)




def save_model(model, name):
path = f'models/{name}.pkl'
joblib.dump(model, path)
print(f"Saved model to {path}")




def save_scaler(scaler):
path = 'models/scaler.pkl'
joblib.dump(scaler, path)
print(f"Saved scaler to {path}")




def plot_and_save_confusion_matrix(y_true, y_pred, labels, name):
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots()
disp.plot(ax=ax)
fig.suptitle(name)
out = f'images/{name.replace(" ", "_").lower()}_confusion_matrix.png'
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f"Saved confusion matrix to {out}")
