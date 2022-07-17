import tensorflow as tf
import os
import shutil
import keras
from keras.layers import Dense,Input,Conv2D,Flatten,Dropout,GlobalMaxPooling2D
from keras.models import Sequential,Model
from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt

def segragate_images(parent_dir,class1='cat',class2='dog'):
    images=os.listdir(parent_dir)
    os.mkdir('data')
    os.chdir('data')
    os.mkdir(class1)
    os.mkdir(class2)
    print(os.curdir)
    os.chdir('..')
    for image in images:
        class_name,_=image.split('_')
        source = os.path.join(parent_dir,image)
        if class_name == class1:
            shutil.move(source,'data/'+class1+'/'+image)
        else:
             shutil.move(source,'data/'+class2+'/'+image)
    return 1

def prepare_data(parent_dir):     # parent_dir-> base directory for all the images
    generator=ImageDataGenerator()
    print(os.getcwd())
    train_it = generator.flow_from_directory(parent_dir, class_mode='binary',target_size=(224,224),batch_size=32)
    # train_ds=preprocess_input(train_it)
    return train_it



def define_VGG16(train_ds,neurons,output_shape):
    base_model = vgg16.VGG16(weights="imagenet", include_top=False, input_shape=(224, 224,3))
    for layer in base_model.layers[:15]:
        layer.trainable = False
    for layer in base_model.layers[15:]:
        layer.trainable = True
    last_layer = base_model.get_layer('block5_pool')
    last_output = last_layer.output

    x = GlobalMaxPooling2D()(last_output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(base_model.input, x)

    # model=Sequential([
    #     base_model,
    #     flatten_layer,
    #     dense_layer_1,
    #     dropout_1,
    #     dense_layer_2,
    #     prediction_layer
    # ])
    return model
def train(parentdir='Datapath',epochs=10):

    trainds=prepare_data(parentdir)
    model=define_VGG16("ndsamd,",100,1)
    model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=["accuracy"])
    hist=model.fit(trainds,epochs=epochs)
    model.save("model.h5")
    # print(model.summary())
    plt.plot(hist.history['accuracy'])



if __name__=='__main__':
    train()
    # segragate_images("D:\RPA\Datapath")
    # generator=ImageDataGenerator()
    # train_it = generator.flow_from_directory('data/', class_mode='binary')
    # prepare_data('')