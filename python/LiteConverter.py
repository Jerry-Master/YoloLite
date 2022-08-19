import tensorflow as tf

# Converting a SavedModel to a TensorFlow Lite model.
saved_model_dir = 'weights/tf/'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model.
with open('weights/yolo.tflite', 'wb') as f:
    f.write(tflite_model)

