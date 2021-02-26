#!/usr/bin/env python
# coding: utf-8

# ### <center>Convolutional Neural Network with Keras

# In[4]:


# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend


# keras doc: https://keras.io/

# In[5]:


class SmallVGGNet:
    
    def build(width, height, channel, classes):
        
        model = Sequential()
        inputShape =(height, width, channel)
        chanDim = -1
        
        # if we are using "channels first", update the input shape and channels dimension
        if backend.image_data_format()== "channels_first":
            inputShape =(channel , height, width)
            chanDim = 1
        
        # CONV => RELU => POOL layer set *3 layer set
        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu")) #Rectified Linear Unit
        
        #Batch Normalization is used to normalize the activations of a given input volume before passing it to the next layer in the network.
        #It has been proven to be very effective at reducing the number of epochs required to train a CNN as well as stabilizing training itself.
        model.add(BatchNormalization(axis = chanDim))
        
        #progressively reducing the spatial size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        #25% of the node connections are randomly disconnected (dropped out) between layers during each training iteration.
        model.add(Dropout(.25)) #concept not to be overlooked
        
        
        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis= chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        #increasing the total number of filters learned from 32 to 64. remainning same dimensions(3x3)
        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        #deeper you go into a CNN (and as your input volume size becomes smaller and smaller) is common practice.
        
        
        #FC => RELU
        # set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))
        
        #softMax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        
        return model
        


# In[ ]:




