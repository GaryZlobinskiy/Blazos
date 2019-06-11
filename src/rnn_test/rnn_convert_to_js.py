import tensorflow as tf
import tensorflowjs as tfjs

look_back = 60
num_features = 5

print("Creating model...")
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(105, return_sequences=True, input_shape=(look_back, num_features)))
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(num_features))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae"])

print("Loading...")
model.load_weights("rnn_model.h5")

print("Converting...")
tfjs.converters.save_keras_model(model, "rnn_model_js")