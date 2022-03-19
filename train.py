import tensorflow as tf
from kale.sdk import step


@step(name="model_training")
def train(model: tf.keras.Model, train_data, test_data, epochs):
    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs
    )

    return model, history.history
