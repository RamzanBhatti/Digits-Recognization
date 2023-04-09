import numpy as np
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import filters
from skimage.filters import prewitt_h,prewitt_v


print("👉👉👉👉Reading Training Data.")
Train_path = "F://1StudyData//5thSemester//AI Assignment 2//data//train"
classes = os.listdir(Train_path)

x_train = []
image_labels = []
for cl in classes:
    x = len(os.listdir(Train_path+"\\"+cl))
    for filename in os.listdir(Train_path+"\\"+cl):
        if any([filename.endswith(x) for x in ['.png']]):
            img = cv2.imread(Train_path+"\\"+cl+"\\"+filename,0)
            resized_img = cv2.resize(img,(200,200))
            prewitt_horizontal = prewitt_h(resized_img)
            x_train.append(prewitt_horizontal)
            if (cl == 'i'):
               image_labels.append(1)
            elif (cl == 'ii'):
               image_labels.append(2)
            elif (cl == 'iii'):
               image_labels.append(3)
            elif (cl == 'iv'):
               image_labels.append(4)
            elif (cl == 'v'):
               image_labels.append(5)
            elif (cl == 'vi'):
               image_labels.append(6)
            elif (cl == 'vii'):
               image_labels.append(7)
            elif (cl == 'viii'):
               image_labels.append(8)
            elif (cl == 'ix'):
               image_labels.append(9)
            elif (cl == 'x'):
               image_labels.append(10)
               
               
X_Train = np.array(x_train)
X_Train = X_Train.reshape(len(X_Train),-1)      
Y_train = np.array(image_labels) 

X_train, X_test, y_train, y_test = train_test_split(X_Train,Y_train,test_size=0.20, random_state=7)

print("👉👉👉👉Training MLP Classifier.")
clf = MLPClassifier(hidden_layer_sizes =(250,150,50),verbose = True, max_iter=1000).fit(X_train, y_train)

print("Prediction on training data.")
y_pred = clf.predict(X_train)
result = np.array(y_pred)
print(accuracy_score(y_train,y_pred)*100,"%")

print("👉👉👉👉Reading validation data.")   
            
Validation_path = "F://1StudyData//5thSemester//AI Assignment 2//data//val"
classes = os.listdir(Validation_path)
x_test = []
image_labels1 = []
for cl in classes:
    x = len(os.listdir(Validation_path+"\\"+cl))
    for filename in os.listdir(Validation_path+"\\"+cl):
        if any([filename.endswith(x) for x in ['.png']]):
            img = cv2.imread(Validation_path+"\\"+cl+"\\"+filename,0)
            resized_img = cv2.resize(img,(200,200))
            prewitt_horizon = prewitt_h(resized_img)
            x_test.append(prewitt_horizon)
            if (cl == 'i'):
               image_labels1.append(1)
            elif (cl == 'ii'):
               image_labels1.append(2)
            elif (cl == 'iii'):
               image_labels1.append(3)
            elif (cl == 'iv'):
               image_labels1.append(4)
            elif (cl == 'v'):
               image_labels1.append(5)
            elif (cl == 'vi'):
               image_labels1.append(6)
            elif (cl == 'vii'):
               image_labels1.append(7)
            elif (cl == 'viii'):
               image_labels1.append(8)
            elif (cl == 'ix'):
               image_labels1.append(9)
            elif (cl == 'x'):
               image_labels1.append(10)

X_Test = np.array(x_test)
X_Test = X_Test.reshape(len(X_Test),-1)
Y_test = np.array(image_labels1)

print("prediction on validation data.")
y_pred = clf.predict(X_Test)
result = np.array(y_pred)
# accuracy = 0
print(accuracy_score(Y_test,y_pred)*100,"%")

