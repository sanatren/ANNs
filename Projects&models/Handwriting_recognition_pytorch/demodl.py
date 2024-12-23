import torch
from torch import nn
from torch.nn.modules import batchnorm
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

epochs = 10
class Ann(nn.Module):
    def __init__(self):
        super(Ann,self).__init__()
        self.flatten = nn.Flatten()
        self.Dense_layers = nn.Sequential(
            nn.Linear(28*28,512) ,#the input attribute is 28*28 because the images in mnist are in 28*28 pixel and setting 512 neurons per dense layer
            nn.ReLU(), #using relu function on layers to reduce vanishing gradient and returning max variations between matrix calculations of bias and weights for best learning and outputs
            nn.Dropout(0.2),batchnorm.BatchNorm1d(512),#applying droputs and batchnormalization for optimization
            nn.Linear(512,10), ##there are total 10 didgits in datasets so 10 outputs are there
            nn.Softmax(dim=1)# output dimension == 1
        )



    def forward(self, x):
        Flatten_data = self.flatten(x)  # Flatten input data
        logic = self.Dense_layers(Flatten_data)  # Pass through the dense layers
        return logic


def download_mnist():
   train_data = datasets.MNIST(
       root = "data",
       train = True,
       download = True,
       transform = ToTensor()
   )
   validation_data = datasets.MNIST(
       root = "data",
       train = False,
       download = True,
       transform = ToTensor()
   )
   return train_data, validation_data

def acc_fn(pred, y):
    return (pred.argmax(1) == y).type(torch.float).sum().item()

def train_a_epoch(model, data_loader, optimizer, loss_fn, acc_fn,device):
    size = len(data_loader.dataset)
    model.train()
    total_loss = 0
    correct = 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # Forward pass and precition on data and finding loss between original and prediction point or (y-y^) for loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass and optimization
        optimizer.zero_grad()#clearing the previous batch gradients for fresh calculation for new batch
        loss.backward()#backpropagation mechanism for updating weights and biases of inputs and nodes while calculation gradients via integration on dependent weights in relation with each other
        optimizer.step()#Updates the model’s parameters (weights) using the gradients calculated in the previous step.

        # Calculate total loss
        total_loss += loss.item()
        correct += acc_fn(pred, y)# Calculate accuracy

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]")#printing loss for each epoch


    avg_loss = total_loss / len(data_loader) # Calculate average loss and accuracy for the epoch
    accuracy = correct / size

    print(f"Train Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {avg_loss:>8f}")#printing acc for a batch

    return avg_loss, accuracy

def train(model, data_loader, optimizer, loss_fn, acc_fn, device, num_epochs=100):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_a_epoch(model, data_loader, optimizer, loss_fn, acc_fn, device)
        print("---------------------------")
    print("Finished training")

if __name__ == "__main__":
    train_data,_ = download_mnist()

    print("data has been downloaded")

    data_loader = DataLoader(train_data, batch_size=64,shuffle=True) #loadind the data in mini batches for fast processing

    #build model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    feed_forward_net =Ann().to(device)

    #build model
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=0.001)

    train(feed_forward_net, data_loader, optimizer, loss_fn, acc_fn, device, num_epochs=epochs)


    torch.save(feed_forward_net.state_dict(),"feed_forward_net.pth")
    print("model saved")

def get_feed_forward_net():
    model = Ann()  # Instantiate your model
    return model
