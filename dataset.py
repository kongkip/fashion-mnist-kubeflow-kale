import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from kale.sdk import step


class Dataset:
    def __init__(self) -> None:
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.image_size = (28, 28)
        self.batch_size = 32
        self.buffer_size = 1000

    @staticmethod
    def normalize(image, label):
        image = tf.cast(image, tf.float32)
        return image / 255.0, label

    @step(name="data_loading")
    def load_data(self, buffer_size=1000):
        """
        Load the fashion dataset and map it to tensorflow datasets for easier processing.
        :return:
        """
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
        train = tf.data.Dataset.from_tensors((train_images, train_labels))
        test = tf.data.Dataset.from_tensors((test_images, test_labels))

        train = train.map(self.normalize, num_parallel_calls=tf.data.AUTOTUNE)\
            .shuffle(self.buffer_size)
        test = test.map(self.normalize, num_parallel_calls=tf.data.AUTOTUNE)

        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        return train, test, class_names
