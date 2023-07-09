```python
#Using Kaggle DataSet
https://www.kaggle.com/datasets/huuthocs/brain-segmentation-for-healthy-and-tumor
```


```python
import pandas as pd
import os
import numpy as np
import cv2
```


```python
path_training = 'train'
path_test = 'test'
```


```python
df = pd.read_csv('brain_multi.csv', names=["file", "result", "col1", "col2", 'col3', 'col4', 'col5'])

dict_test = dict(zip(df.file,df.result))

```


```python
def read_images_from_path(path):
    print(path)
    all_images = []
    y = []
    WIDTH = 640
    HEIGHT = 640
    for image_path in os.listdir(path):
      if image_path.endswith('.csv'):
        continue
      #print(path_training+'/'+image_path)  
      image = cv2.imread(path+'/'+image_path,cv2.IMREAD_GRAYSCALE)
      if image is None:
        continue
      if image_path not in dict_test:
        continue
      res = 0 if dict_test[image_path] == 'normal' else 1  
      img = cv2.resize(image, (WIDTH, HEIGHT))
      normalized_image = img / 255.0  
      all_images.append(normalized_image)
      y.append(res)
    x = np.array(all_images)
    return (x, y) 
```


```python

```


```python
def read_images_from_path_test(path):
    print(path)
    all_images = []
    y = []
    WIDTH = 640
    HEIGHT = 640
    for image_path in os.listdir(path):
      #print(path_training+'/'+image_path)  
      image = cv2.imread(path+'/'+image_path,cv2.IMREAD_GRAYSCALE)
      if image is None:
        continue
      if 'Not-Cancer' in image_path:
        y.append(0)
      else:
        y.append(1)  
      img = cv2.resize(image, (WIDTH, HEIGHT))
      normalized_image = img / 255.0  
      all_images.append(normalized_image)
    x = np.array(all_images)
    return (x, y) 
```


```python
X_train, y_train = read_images_from_path(path_training)



```

    train



```python
X_test,y_test  = read_images_from_path_test(path_test)
```

    test



```python
from keras.utils import to_categorical
```

    2023-07-09 03:26:23.112305: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-07-09 03:26:23.114326: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-07-09 03:26:23.159137: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-07-09 03:26:23.159681: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-07-09 03:26:23.814442: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



```python
y_train = np.array(y_train)
```


```python
y_train.shape
```




    (5744,)




```python
y_test = np.array(y_test)
```


```python
y_test.shape
```




    (383,)




```python
n_classes = 2
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)
```

    Shape before one-hot encoding:  (5744,)
    Shape after one-hot encoding:  (5744, 2)



```python
y_test
```




    array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
           0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1,
           1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
           0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
           1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0,
           1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1,
           0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
           0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0,
           0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
           1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
           0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,
           0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1,
           1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
           0, 1, 1, 0, 0, 0, 1, 0, 0])




```python
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
#from keras.utils import np_utils
# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer
model.add(Conv2D(30, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(640,640,1)))
model.add(MaxPool2D(pool_size=(1,1)))
```


```python
# flatten output of conv
model.add(Flatten())
```


```python
# hidden layer
model.add(Dense(100, activation='relu'))
# output layer
model.add(Dense(2, activation='softmax'))
```


```python
# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

```


```python
# training the model for 10 epochs
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
```

    Epoch 1/10
    45/45 [==============================] - 330s 7s/step - loss: 21.9929 - accuracy: 0.7587 - val_loss: 1.6744 - val_accuracy: 0.7911
    Epoch 2/10
    45/45 [==============================] - 324s 7s/step - loss: 0.3256 - accuracy: 0.9284 - val_loss: 0.4154 - val_accuracy: 0.9138
    Epoch 3/10
    45/45 [==============================] - 325s 7s/step - loss: 0.0638 - accuracy: 0.9779 - val_loss: 0.2812 - val_accuracy: 0.9399
    Epoch 4/10
    45/45 [==============================] - 324s 7s/step - loss: 0.0296 - accuracy: 0.9896 - val_loss: 0.2221 - val_accuracy: 0.9608
    Epoch 5/10
    45/45 [==============================] - 324s 7s/step - loss: 0.0077 - accuracy: 0.9977 - val_loss: 0.2196 - val_accuracy: 0.9556
    Epoch 6/10
    45/45 [==============================] - 325s 7s/step - loss: 0.0035 - accuracy: 0.9998 - val_loss: 0.2643 - val_accuracy: 0.9530
    Epoch 7/10
    45/45 [==============================] - 324s 7s/step - loss: 0.0023 - accuracy: 1.0000 - val_loss: 0.2313 - val_accuracy: 0.9556
    Epoch 8/10
    45/45 [==============================] - 326s 7s/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.2421 - val_accuracy: 0.9582
    Epoch 9/10
    45/45 [==============================] - 325s 7s/step - loss: 0.0012 - accuracy: 1.0000 - val_loss: 0.2720 - val_accuracy: 0.9530
    Epoch 10/10
    45/45 [==============================] - 327s 7s/step - loss: 9.2953e-04 - accuracy: 1.0000 - val_loss: 0.2730 - val_accuracy: 0.9530





    <keras.src.callbacks.History at 0x7f67c45fe0d0>




```python
respopnse_test = model.predict(X_test)
```

    12/12 [==============================] - 9s 736ms/step



```python
#Check the first element the current vs the actual predicted value
```


```python
#Predicted as tumor
respopnse_test[0]
```




    array([8.0367880e-07, 9.9999917e-01], dtype=float32)




```python
#Actual is tumor
Y_test[0]
```




    array([0., 1.], dtype=float32)




```python

```
