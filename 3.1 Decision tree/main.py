def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

my_data=pd.read_csv('drug200.csv')
my_data.head()

# khai bao X la cac bien du lieu, y la muc tieu
# xoa cac cot bao gom ten muc tieu khong chua muc tieu va khong o dang so

X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# mot so thuoc tinh nhu Sex hay bp la cac bien phan loai, sklearn decision tree khong the xu ly duoc nen phai dung thu vien pandas de chuyen tu bien phan loai sang dummy (0/1) hay indicator 

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 
# X[:,1] co nghia la cat mang, lay het tat ca cac hang (:) nhung giu lai cot thu hai (1)


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]

y = my_data["Drug"]
y[0:5]

from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)
#cai nay se chia du lieu thanh cac set de train va test nhu tren ta co test_size = 0.3 va so lan shuffle du lieu random_state=3)


# tao cay phan loai duoc goi la drugTree, tao mot cai tieu chi ten la entropy de xem IG cua moi node
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
drugTree.fit(X_trainset,y_trainset) # fit data voi training feature X_trainset va training y_trainset


# du doan tren test data va luu no vao mot cay ten predTree

predTree = drugTree.predict(X_testset)

print (predTree [0:5])
print (y_testset [0:5])


# do do chuan xac cua model
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# ve cay
tree.plot_tree(drugTree)
plt.show()