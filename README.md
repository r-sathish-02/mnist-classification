# EX-03 Convolutional Deep Neural Network for Digit Classification

### Aim:
<table>
<tr>
<td width=80%>
  
To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.


</td> 
<td valign=top>


</td>
</tr> 
</table>

### Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.The MNIST dataset is a collection of handwritten digits.The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.


### Neural Network Model

<img height=15% src="https://github.com/user-attachments/assets/daebdd33-6aba-41a0-bcb1-87145a520591">


### DESIGN STEPS
- **Step 1:** Import tensorflow and preprocessing libraries,Download and load the dataset
- **Step 2:** Scale the dataset between it's min and max values, Using one hot encode, encode the categorical values
- **Step 3:** Split the data into train and test, Build the convolutional neural network model
- **Step 4:** Train the model with the training data, Plot the performance plot
- **Step 5:** Evaluate the model with the testing data, Fit the model and predict the single input
### Program:
```Python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(xtrain, ytrain),(xtest, ytest)=mnist.load_data()
xtrainS=xtrain/255.0
xtestS=xtest/255.0
ytren = utils.to_categorical(ytrain,10)
yteen = utils.to_categorical(ytest,10)
xtrainS = xtrainS.reshape(-1,28,28,1)
xtestS = xtestS.reshape(-1,28,28,1)
model = models.Sequential([layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu'),
        layers.MaxPool2D(pool_size=(3, 3)), layers.Flatten(),
        layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'), layers.Dense(10, activation='softmax')])
model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(xtrainS,ytren,epochs=15,batch_size=256,validation_data=(xtestS,yteen))
metrics = pd.DataFrame(model.history.history)
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()
ypred = np.argmax(model.predict(xtestS), axis=1)
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
img = image.load_img('image.png')
tensor_img = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(tensor_img,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
plt.title('Sathish R  212222230138')
np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```
### Output:

### Training Loss, Validation Loss Vs Iteration Plot

<img height=15% width=48% src="https://github.com/user-attachments/assets/54dd5ccf-95f9-4592-a86c-031f3106443a"><img height=15% width=48% src="https://github.com/user-attachments/assets/cedebe32-8645-4bfb-9209-4b05542950ad">


<table>
<tr>
<td width=48%>
  
### Classification Report
![image](https://github.com/user-attachments/assets/96a8acd4-0633-4c16-b66b-b46a3d39cda4)
</td> 
<td valign=top>

### Confusion Matrix
![image](https://github.com/user-attachments/assets/45bb2923-cfcf-4eec-850e-9bce8263774d)</td>
</tr> 
</table>


### New Sample Data Prediction

<img height=20% src="https://github.com/user-attachments/assets/7674fcf8-5c45-4503-a2f6-b7944394a5ed">


### RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
