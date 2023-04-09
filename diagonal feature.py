
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from sklearn.metrics import accuracy_score
from matplotlib.patches import Rectangle
from PIL import Image 
import glob
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
            resized_image = cv2.resize(img,(200,200))
            x_train.append(resized_image)
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
X = X_Train.reshape(len(X_Train), -1)
X_Train = np.fliplr(X).diagonal()
X_Train = X_Train.reshape(len(x_train),1)
Y_Train = np.array(image_labels)

X_train, X_test, y_train, y_test = train_test_split(X_Train,Y_Train,test_size=0.20,random_state=25)
print("👉👉👉👉Training MLP Classifier.")
clf = MLPClassifier(hidden_layer_sizes = (100,50,10),verbose=True, max_iter=1000).fit(X_train, y_train)

print("Prediction on training data.")
result = clf.predict(X_train)
result = np.array(result)
accuracy = 0
for i in range(len(result)):
    accuracy += 1 if result[i] == y_train[i] else 0
print('Result',(accuracy/len(result))*100,'%')

print("👉👉👉👉Reading validation data.")  
Train_path = "F://1StudyData//5thSemester//AI Assignment 2//data//val"
classes = os.listdir(Train_path)

x_test = []
image_labels = []
for cl in classes:
    x = len(os.listdir(Train_path+"\\"+cl))
    for filename in os.listdir(Train_path+"\\"+cl):
        if any([filename.endswith(x) for x in ['.png']]):
            img = cv2.imread(Train_path+"\\"+cl+"\\"+filename,0)
            resized_image = cv2.resize(img,(200,200))
            x_test.append(resized_image)
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
            
             

X_test = np.array(x_test)
X = X_test.reshape(len(X_test), -1)
x_test = np.fliplr(X).diagonal()
x_test = x_test.reshape(len(x_test),1)
y_test = np.array(image_labels)

print("prediction on validation data.")
result = clf.predict(x_test)
result = np.array(result)
accuracy = 0
for i in range(len(result)):
    accuracy += 1 if result[i] == y_test[i] else 0
print('Result',(accuracy/len(result))*100,'%')

