#Loading all libraries
import os
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

#Setting device to CUDA
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#Applying Image Transforms
transformer=transforms.Compose([
    transforms.Resize((108,108)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],
                        [0.5,0.5,0.5])
])


#Since there is an upload limit of 100mb, we have uploaded only a small subset of 
#the actual NORB Dataset that we trained on so that the code can be tested here.

#Path for training and testing directory
train_path= os.path.join(os.getcwd(), "Sample Dataset", "Training Data")
test_path= os.path.join(os.getcwd(), "Sample Dataset", "Testing Data")

#Loading Training and Testing Data
train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=64, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32, shuffle=True
)


#Printing all classes in the dataset
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)

#Setting Number of epochs
num_epochs=10

class LinearNet(nn.Module):
    def __init__(self, num_classes=6):
        super(LinearNet, self).__init__()
        self.network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(108*108*3, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes))  # output size depends on the number of classes in your dataset
                      
    def forward(self, input):
        return self.network(input)
    
model=LinearNet(num_classes=6).to(device)

#Defining Optmizer and loss function
optimizer=Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
loss_function=nn.CrossEntropyLoss()

#Confirming the size of training and testing images
train_count=len(glob.glob(train_path+'/**/*.jpg'))
test_count=len(glob.glob(test_path+'/**/*.jpg'))
print('Numer of training images:', train_count)
print('Numer of testing images:', test_count)

#Model training and testing for a set number of epochs
#Creating lists to be used for plotting graphs
loss_values = []
train_acc_values = []
test_acc_values = []

#Lists for confusion matrix
true_labels = []
predicted_labels = []

for epoch in range(num_epochs):
    
    #Training on training dataset
    model.train()
    train_accuracy=0.0
    train_loss=0.0
    
    for i, (images,labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        optimizer.zero_grad()
        
        outputs=model(images)
        loss=loss_function(outputs,labels)
        loss.backward()
        optimizer.step()
        
        
        train_loss+= loss.cpu().data*images.size(0)
        _,prediction=torch.max(outputs.data,1)
        
        train_accuracy+=int(torch.sum(prediction==labels.data))
        
    train_accuracy=train_accuracy/train_count
    train_loss=train_loss/train_count
    loss_values.append(train_loss)
    train_acc_values.append(train_accuracy)
    print('Epoch: '+str(epoch)+' Train Loss: '+str(train_loss)+' Train Accuracy: '+str(train_accuracy))
    
    # Evaluation on testing dataset
    model.eval()
    test_accuracy=0.0
    for i, (images,labels) in enumerate(test_loader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())
            
        outputs=model(images)
        _,prediction=torch.max(outputs.data,1) 
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy=test_accuracy/test_count
    test_acc_values.append(test_accuracy)
    print(' Test Accuracy: '+str(test_accuracy))
    
    # Store the true labels and predicted labels for the last epoch
    if epoch == num_epochs - 1:
        # Iterate over the test data
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images=Variable(images.cuda())
                labels=Variable(labels.cuda())
            # Get the predicted labels for the images
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Convert the labels and predicted labels to numpy arrays
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

print('save training results')
print('Losses for each epoch: '+str(loss_values))
print('Training Accuracy for each epoch: '+str(train_acc_values))

print('save testing results')
print('Testing accuracy for each epoch: '+str(test_acc_values))

#Saving the model
#torch.save(model.state_dict(),'best_checkpoint.model')


# Confusion Matrix
true_labels = list(map(int, true_labels))
predicted_labels = list(map(int, predicted_labels))
conf_matrix = confusion_matrix(true_labels, predicted_labels)
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_matrix, annot=True, fmt='.0%',
            cbar_kws={'format': mtick.PercentFormatter(xmax=1.0, decimals=0)},
            xticklabels=[i+1 for i in range(6)], 
            yticklabels=[i+1 for i in range(6)])
plt.show()


# Plotting the loss values
#loss_values = [loss.item() for loss in loss_values]
epoch = range(1, len(train_acc_values)+1)
plt.plot(epoch, loss_values, '-o', color='red', label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, 11))
plt.show()

#Converting accuracies to percentage
train_acc_values_percent = [i * 100 for i in train_acc_values]
test_acc_values_percent = [i * 100 for i in test_acc_values]
print(train_acc_values_percent)
print(test_acc_values_percent)

# Plotting the training accuracy values
plt.plot(epoch, train_acc_values_percent, '-o', color='blue', label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy in %')
plt.xticks(range(0, 11))
plt.show()

# Plotting the testing accuracy values
plt.plot(epoch, test_acc_values_percent, '-o', color='blue', label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Testing Accuracy in %')
plt.xticks(range(0, 11))
plt.show()

#Plotting testing vs training accuracy
plt.plot(epoch, train_acc_values_percent, '-o', color='red', label='Training accuracy')
plt.plot(epoch, test_acc_values_percent, '-o', color='blue', label='Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy in %')
plt.xticks(range(0, 11))
plt.yticks(range(45,100, 5))
plt.legend()
plt.title('Training vs Testing accuracy of LNN model')
plt.show()
