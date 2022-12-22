import onnx
onnx_model_path = 'test.onnx'
tf_model_path = 'onnx2pb.pb'
onnx_model = onnx.load(onnx_model_path)

from onnx_tf.backend import prepare

tf_rep = prepare(onnx_model)

tf_rep.export_graph(tf_model_path)

import tensorflow as tf

model = tf.saved_model.load(tf_model_path)
model.trainable = False

# input_tensor = tf.random.uniform([batch_size, channels, height, width])
# out = model(**{'input': input_tensor})

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()

# Save the model
# with open(tflite_model_path, 'wb') as f:
#     f.write(tflite_model)