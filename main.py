import tensorflow as tf
import numpy as np

# Data (numbers)
x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([0, 1, 0, 1, 0, 1], dtype=float)  # 0 = odd, 1 = even

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

# Train
model.fit(x, y, epochs=500, verbose=0)

# Test
print("Prediction for 10:", model.predict([10]))
print("Prediction for 7:", model.predict([7]))