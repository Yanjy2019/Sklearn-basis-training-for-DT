import numpy as np
from sklearn import datasets
d=datasets.load_breast_cancer()
x=d.data
y=d.target
y=y.reshape(-1,1)
print(x)
print(x.shape)
import matplotlib.pyplot as plt
plt.figure()
x1=np.linspace(0,10,100)
y1=np.sin(x1**2-2)
plt.plot(x1,y1,"r")
plt.show()
from sklearn.tree import DecisionTreeClassifier
D=DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=666)
D.fit(x_train,y_train)
print(D.score(x_test,y_test))
from sklearn import datasets
d1=datasets.load_boston()
x2=d1.data
y2=d1.target
