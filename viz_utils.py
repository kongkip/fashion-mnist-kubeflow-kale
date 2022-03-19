import matplotlib.pyplot as plt
from kale.sdk import step

@step(name="plot_results")
def plot_results(history, epochs):
    """
    Generate plots for the training losses and accuracies
    :param history:
    :param epochs:
    :return:
    """

    def _viz_acc():
        train_accs = history["accuracy"]
        test_accs = history["val_accuracy"]
        iterations = [i + 1 for i in range(int(epochs))]

        plt.plot(iterations, train_accs, color="blue", label="Training Accuracy")
        plt.plot(iterations, test_accs, color="red", label="Testing Accuracy")
        plt.legend()
        plt.show()

    def _viz_loss():
        train_losses = history["loss"]
        test_losses = history["val_loss"]
        iterations = [i + 1 for i in range(int(epochs))]

        plt.plot(iterations, train_losses, color="blue", label="Training Loss")
        plt.plot(iterations, test_losses, color="red", label="Testing Loss")
        plt.legend()
        plt.show()

    _viz_acc()
    _viz_loss()

@step(name="plot_predictions")
def plot_predictions(model, test_data, class_names):
    preds = model.predict(test_data)
    for image, label in test_data:
        for i in range(10):
            predicted_label = preds[i].argmax()
            if predicted_label == label[i]:
                color = 'blue'
            else:
                color = 'red'
            plt.imshow(image[i], cmap=plt.cm.binary)
            plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                                 100 * preds[i][predicted_label].max(),
                                                 class_names[label[i]]), color=color)
    plt.show()
