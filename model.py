from multiprocessing.reduction import steal_handle
import tensorflow as tf
from kale.sdk import step

@step(name="model_creation")
def get_model():
    """
    Define a simple sequential model
    :return:
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print(model.summary())
    return model
