import json
import numpy as np
import cv2
#import sklearn.metrics as sklm
import helper
from keras.applications.vgg16 import VGG16
#from keras.preprocessing import image
#from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model, load_model
#from keras.datasets import mnist

from keras import backend as K
img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)

# the model
def pretrained_model(img_shape, num_classes, layer_type):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    #model_vgg16_conv.summary()
    
    #Create your own input format
    keras_input = Input(shape=img_shape, name = 'image_input')
    
    #Use the generated model 
    output_vgg16_conv = model_vgg16_conv(keras_input)
    
    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation=layer_type, name='fc1')(x)
    x = Dense(4096, activation=layer_type, name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    #Create your own model 
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return pretrained_model


def train_model():
    sett = helper.create_test_set("ROOT\\DATASET\\file_indices.json", 1)
    for idx, train_set, test_set in sett:
        x_train = []
        x_test = []
        
        # loading the data
        for f in train_set:
            _img = cv2.resize(cv2.imread(f['path']), (100,100))
            x_train.append(_img)
        x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')
        y_train = [float(x['label_num']) for x in train_set]
        for f in test_set:
            _img = cv2.resize(cv2.imread(f['path']), (100,100))
            x_test.append(_img)
        x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')
        y_test = [float(x['label_num']) for x in test_set]
        
        # training the model
        model = pretrained_model(x_train.shape[1:], len(set(y_train)), 'relu')
        hist = model.fit(x_train, y_train, epochs=1000, verbose=1)
        
        #save indices
        with open("ROOT\\train_files.json", 'w') as outfile:
            json.dump(train_set, outfile)
            
        with open("ROOT\\test_files.json", 'w') as outfile:
            json.dump(test_set, outfile)
        
        #save model
        model.save("ROOT\\model.h5")

def test_model():
    model = load_model("ROOT\\model.h5")
    #test acc
    with open("ROOT\\test_files.json") as f:
        test_set = json.load(f)
    x_test = []
        
    for f in test_set:
        _img = cv2.resize(cv2.imread(f['path']), (100,100))
        x_test.append(_img)
    x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')
    y_test = [x['label_num'] for x in test_set]
    
    lb = model.predict(x_test)
    
    true = 0
    idx = 0
    for x in lb:
        for y in x:
            if y == y_test[idx]:
                true += 1
        idx += 1
    
    print("ACC = " + str(true / len(y_test)))
    
train_model()
test_model()
