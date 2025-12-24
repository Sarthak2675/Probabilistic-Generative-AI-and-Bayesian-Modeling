### Add necessary imports ###
from torch.utils.data import DataLoader
import torch
import math
from models import SimpleNN, CNN

'''
hyperparam_space = {
        0. 'layer_size' : [100,200,300,400,500],
        1. 'epochs' : torch.arange(1,11, 1),
        2. 'log_lr': torch.arange(-5,-1, 0.5),
        3. 'batch_size': [16,32,64,128,256],
        4. 'dropout_rate': torch.arange(0,0.51,0.05),
        5. 'log_weight_decay' : torch.arange(-6,-1.5,0.5)
    }
    indexing according to the above order.
'''


def train_and_test_NN(datasets, hyperparameters, seed=42):
    """
    Train and test a Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    ## converting hyperparams from tensor to python ints/floats
    hyperparams = hyperparameters.clone()
    hyperparams[2] = torch.pow(10,hyperparams[2])
    hyperparams[5] = torch.pow(10,hyperparams[5])
    hyperparams = [p.item() for p in hyperparams]
    hyperparams[0] = int(hyperparams[0])
    hyperparams[1] = int(hyperparams[1])
    hyperparams[0] = int(math.ceil( hyperparams[0]/100.0 )*100)
    hyperparams[3] = int(math.pow(2,int(math.ceil(hyperparams[3]))))
    
    
    print(f'Selected Hyperparameters: {hyperparams}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset, validation_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=hyperparams[3], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=hyperparams[3], shuffle=False)

    ### Implement training loop here ###
    model = SimpleNN(input_size=train_dataset[0][0].flatten().shape[0], hidden_size=hyperparams[0], num_classes=10, dropout_rate=hyperparams[4]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams[2], weight_decay=hyperparams[5])

    for i in range(hyperparams[1]):
        model.train()
        avg_loss = 0
        j = 0
        for inputs, labels in train_loader:
            ## changing shape of inputs from (batch_size, 1, 28, 28) to (batch_size, 784)
            inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            avg_loss += loss
            j+=1
            optimizer.step()
        print(f"Epoch[{i}/{hyperparams[1]}]: loss = {avg_loss/j}")

    ### Implement validation loop here ###
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    return accuracy

def train_and_test_CNN(datasets, hyperparameters, seed=42):
    """
    Train and test a Convolutional Neural Network model
    datasets: tuple of (train_dataset, test_dataset)
    hyperparams: data structure containing hyperparameters like learning rate, epochs, etc.

    Returns:
    accuracy: accuracy on validation dataset
    """
    hyperparams = hyperparameters.clone()
    hyperparams[2] = torch.pow(10,hyperparams[2])
    hyperparams[5] = torch.pow(10,hyperparams[5])
    hyperparams = [p.item() for p in hyperparams]
    hyperparams[0] = int(hyperparams[0])
    hyperparams[1] = int(hyperparams[1])
    hyperparams[0] = int(math.ceil( hyperparams[0]/100.0 )*100)
    hyperparams[3] = int(math.pow(2,int(math.ceil(hyperparams[3]))))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset, validation_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=hyperparams[3], shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=hyperparams[3], shuffle=False)

    ### Implement training loop here ###
    model = CNN(num_classes=10, dropout_rate=hyperparams[4]).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams[2], weight_decay=hyperparams[5])

    for _ in range(hyperparams[1]):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    ### Implement validation loop here ###
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total 
    return accuracy
