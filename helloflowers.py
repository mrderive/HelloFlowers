import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

train_ds = tf.keras.preprocessing.image_dataset_from_directory("flower_photos", image_size=(240, 240), batch_size=32, subset="training", validation_split=0.2, seed=123)
val_ds = tf.keras.preprocessing.image_dataset_from_directory("flower_photos", image_size=(240, 240), batch_size=32, subset="validation", validation_split=0.2, seed=123)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

filters = 100
rate = 0.4
kernel_size = 4

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(240, 240, 3)))
model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.1))
model.add(tf.keras.layers.experimental.preprocessing.RandomZoom(0.1))
model.add(tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)) 

model.add(tf.keras.layers.Conv2D(kernel_size=kernel_size, strides=1, padding="valid", filters=filters, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Dropout(rate=rate))
model.add(tf.keras.layers.Conv2D(kernel_size=kernel_size, strides=1, padding="valid", filters=filters, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Dropout(rate=rate))
model.add(tf.keras.layers.Conv2D(kernel_size=kernel_size, strides=1, padding="valid", filters=filters, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Dropout(rate=rate))
model.add(tf.keras.layers.Conv2D(kernel_size=kernel_size, strides=1, padding="valid", filters=filters, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Dropout(rate=rate))
model.add(tf.keras.layers.Conv2D(kernel_size=kernel_size, strides=1, padding="valid", filters=filters, activation="relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2, padding="valid"))
model.add(tf.keras.layers.Dropout(rate=rate))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=5, activation=None))

optimizer = tf.keras.optimizers.Adam(learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay([100], [0.00001, 0.00001]))

model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=["accuracy"])

model.load_weights("flower_weights")
model.evaluate(x=val_ds)
# model.fit(x=train_ds, epochs=999, validation_data=val_ds, verbose=1)
