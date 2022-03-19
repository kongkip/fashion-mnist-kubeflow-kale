from kale.sdk import step


@step(name="model_evaluation")
def evaluate(model, test_data):
    validation_acc_loss = model.evaluate(test_data)

    print(f"Model Evaluation Accuracy : {validation_acc_loss[1]:.4f}")
    print(f"Model Evaluation Loss : {validation_acc_loss[0]:.4f}")
