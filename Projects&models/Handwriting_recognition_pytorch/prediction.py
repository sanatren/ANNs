#here we will learn about the how to use the trained model and do predictions based on our models required inputs and targets# Importing necessary modules
import torch
from demodl import Ann, download_mnist

class_mapping = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input.unsqueeze(0))  # Add batch dimension
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # Load the model
    model = Ann()
    state_dict = torch.load("feed_forward_net.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Load MNIST validation dataset
    _, validation_data = download_mnist()
    # Find a sample with label 0 in the validation dataset

    for i, (input, target) in enumerate(validation_data):
        if target == 1:
            print(f"Found a sample with label 0 at index {i}")
            break

    # Get a sample from the validation dataset for inference
    input, target = validation_data[0][0], validation_data[0][1]

    # Make an inference
    predicted, expected = predict(model, input, target, class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
