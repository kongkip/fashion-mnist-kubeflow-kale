from kale.sdk import pipeline

from dataset import Dataset
from evaluate import evaluate
from model import get_model
from train import train
from viz_utils import plot_results, plot_predictions


@pipeline(name="fashion-mnist-classification", experiment="fashion-mnist")
def main(epochs):
    # load data
    dataset = Dataset()
    train_data, test_data = dataset.load_data()

    # get the model
    model = get_model()

    # train the model
    trained_model, history = train(model, train_data, test_data, epochs=EPOCHS)

    # evaluate the model
    evaluate(trained_model, test_data)

    # plot results and predictions
    plot_results(history, epochs=EPOCHS)
    plot_predictions(trained_model, test_data, dataset.class_names)


if __name__ == '__main__':
    EPOCHS = 10
    main(EPOCHS)