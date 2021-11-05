import tensorflow as tf
from tensorflow import keras
input_model = 'catdog_mobilenetv2.h5'
output_model = 'catdog_mobilenetv2.tflite'

model = keras.models.load_model(input_model, compile=False)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(output_model, 'wb') as f:
    f.write(tflite_model)
