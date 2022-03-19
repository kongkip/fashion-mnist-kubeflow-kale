from kale.sdk import pipeline

from dataset import Dataset
from evaluate import evaluate
from model import get_model
from train import train
from viz_utils import plot_results, plot_predictions


@pipeline(name="fashion-mnist-classification", experiment="fashion-mnist")
def ml_pipeline(epochs):
    """
    run the kubeflow pipeline
    """

    # load data
    dataset = Dataset()
    train_data, test_data, class_names = dataset.load_data()

    # get the model
    model = get_model()

    # train the model
    trained_model, history = train(model, train_data, test_data, epochs=epochs)

    # evaluate the model
    evaluate(trained_model, test_data)

    # plot results and predictions
    plot_results(history, epochs=epochs)
    plot_predictions(trained_model, test_data, class_names)


if __name__ == '__main__':
    ml_pipeline(EPOCHS=10)
