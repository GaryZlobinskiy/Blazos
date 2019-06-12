import tensorflow as tf
import tensorflowjs as tfjs

look_back = 240
num_features = 5

print("Creating model...")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(120, input_dim=num_features*look_back))
model.add(tf.keras.layers.Dense(60))
model.add(tf.keras.layers.Dense(60))
model.add(tf.keras.layers.Dense(num_features))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

print("Loading...")
model.load_weights("rnn_model.h5")

print("Converting...")
tfjs.converters.save_keras_model(model, "rnn_model_js")