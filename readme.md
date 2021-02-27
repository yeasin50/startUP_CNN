## Start with Keras, Deep Learning, and Python   

Download animal dataset from [kaggle](https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda)  


</br>   

### Training Convolutional Neural Network with Keras
It is named += cnn 
- CNN trainner [train_vgg](https://github.com/yeasin50/startUP_CNN/blob/master/train_vgg.py.ipynb)
- CNN helper [trainerCNN_SmallVGGNe](https://github.com/yeasin50/startUP_CNN/blob/master/trainerCNN_SmallVGGNet.py) or [train_vgg.py](https://github.com/yeasin50/startUP_CNN/blob/master/train_vgg.py.ipynb)
  
We are using this script inside trainerCNN_SmallVGGNe to train model

---
## Train Score
With just 10 Epochs 😅, train with around 70


![CNN train Score](https://github.com/yeasin50/startUP_CNN/blob/master/trainOutput/scores_CNNVGGTest.png)


Also you can use this model just goto [trainOutput Folder](https://github.com/yeasin50/startUP_CNN/tree/master/trainOutput)

---- 
Notice here, we used  categorical_crossentropy because of multi-class
```
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
```
Categorical cross-entropy is used as the loss for nearly all networks trained to perform classification. The only exception is for 2-class classification where there are only two possible class labels. In that event you would want to swap out `"categorical_crossentropy"` for "binary_crossentropy" .