import tensorflow as tf

# Load Model
def classifier(path=r'models/model_training/model.h5'):
        model =tf.keras.models.load_model(path,custom_objects={'BatchNorm': tf.keras.layers.BatchNormalization})
        return model



#for this I have to create data validater


