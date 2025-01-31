import numpy as nm
import pandas as pd
from sklearn.svm import SVC
svm_model=SVC(kernel='linear')
df=pd.read_csv("E:\\diabetes.csv")
df.head()
df.info()
x=df.iloc[:, [1,2,3,4,5,6,7]].values
y=df.iloc[:, 8].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.20,random_state=0)
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
svc_classifier = SVC(kernel='rbf', random_state=0,verbose=True)
dt_classifier = DecisionTreeClassifier ( random_state=0)
k_classifier = KNeighborsClassifier(n_neighbors=3)

from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import matplotlib as plt

##SVC Algorithms
svc_classifier.fit(x_train, y_train)
svc_y_pred= svc_classifier.predict(x_test)

svc_acc = accuracy_score(y_test,svc_y_pred)
svc_f1= f1_score(y_test,svc_y_pred)
svc_recall_sc = recall_score(y_test,svc_y_pred)
svc_pre_sc = precision_score(y_test,svc_y_pred)

# plt.bar([1,1,1,1],[svc_acc,svc_f1,svc_recall_sc,svc_pre_sc])
##Decision tree Algorithms
dt_classifier.fit(x_train, y_train)
dt_y_pred= dt_classifier.predict(x_test)

dt_acc = accuracy_score(y_test,dt_y_pred)
dt_f1= f1_score(y_test,dt_y_pred)
dt_recall_sc = recall_score(y_test,dt_y_pred)
dt_pre_sc = precision_score(y_test,dt_y_pred)

##KNearNeighbours tree Algorithms
k_classifier.fit(x_train, y_train)
k_y_pred= k_classifier.predict(x_test)

k_acc = accuracy_score(y_test,k_y_pred)
k_f1= f1_score(y_test,k_y_pred)
k_recall_sc = recall_score(y_test,k_y_pred)
k_pre_sc = precision_score(y_test,k_y_pred)

print([svc_acc,svc_f1,svc_recall_sc,svc_pre_sc])
print([dt_acc,dt_f1,dt_recall_sc,dt_pre_sc])
print([k_acc,k_f1,k_recall_sc,k_pre_sc])

import matplotlib.pyplot as plt
import numpy as np



# Define the categories
metrics = ['Accuracy', 'F1 Score', 'Recall', 'Precision']

# Create a numpy array with the performance data
data = np.array([[svc_acc, svc_f1, svc_recall_sc, svc_pre_sc],
                 [dt_acc, dt_f1, dt_recall_sc, dt_pre_sc],
                 [k_acc, k_f1, k_recall_sc, k_pre_sc]])

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define the x-axis positions for the bars
x = np.arange(len(metrics))  # The label locations
width = 0.2  # The width of the bars

# Plot bars for each algorithm
ax.bar(x - width, data[0], width, label='SVC')
ax.bar(x, data[1], width, label='Decision Tree')
ax.bar(x + width, data[2], width, label='KNN')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Algorithms Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Display the chart
plt.tight_layout()
plt.show()
