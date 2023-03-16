import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_data_loader(train):
    """
    TODO: implement this function.
    
    INPUT: 
        An optional boolean argument (default value is True for training dataset)
    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    
    custom_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if (train == True):
        train_set=datasets.FashionMNIST('./data', train = True, download = True,transform = custom_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
        return train_loader
    if (train == False):
        test_set=datasets.FashionMNIST('./data', train = False, transform = custom_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64)
        return test_loader

def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 10, bias=True),
        )
    return model

def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(T):
        correct = 0
        total = 0  #Total data number
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader, 0):
            # data is a batch of images(inputs) and labels
            images,labels = data 

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels) #Calculate how wrong
            loss.backward()
            opt.step()

            running_loss += loss.item()
    
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print(f'Train Epoch: {epoch} Accuracy: {correct}/60000 ({correct/600:.2f}%) Loss: {running_loss * T / total:.3f}')
        running_loss = 0.0

def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    #model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels) #Calculate how wrong
        loss.backward()
        running_loss += loss.item()

    if show_loss == True:
        print(f'Average loss: {running_loss/ total :.4f}')
        print(f'Accuracy: {correct * 100 /total :.2f} %')
    if show_loss == False:
        print(f'Accuracy: {correct * 100/ total:.2f} %')
    
def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    
    prob = model(test_images[index])
    _, predicted = torch.max(prob.data, 1)
    prob = F.softmax(prob, dim=1)
    max_prob = prob.argmax(dim=1)

    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    for images in prob:
        sorted, indices = images.sort(descending=True)
        for i in range(3):
            print('{}: {:.2f}%'.format(class_names[i], (sorted[i] * 100)))

if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
