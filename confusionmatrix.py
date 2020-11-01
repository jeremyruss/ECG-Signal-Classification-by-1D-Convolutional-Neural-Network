from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import json

with open('./data.json') as f:
    data = json.load(f)

df = data["data"]

y_true = []
y_pred = []

for value in df:
    y_t = value[187]
    y_p = value[188]
    y_true.append(y_t)
    y_pred.append(y_p)

cm = confusion_matrix(y_true, y_pred, normalize='true')
ax = sn.heatmap(cm, cmap='cividis', annot=True)
plt.xlabel('xlabel')
plt.ylabel('ylabel')
plt.show()
