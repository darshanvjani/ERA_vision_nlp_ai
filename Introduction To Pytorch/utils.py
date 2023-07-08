# !pip install torchsummary

from torchsummary import summary
import torch
import model
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def check_cuda():
  if torch.cuda.is_available():
    print("CUDA Available?", torch.cuda.is_available())
    return True
  else:
    return False



def dataset(transformation=False):
  # Train data transformations
  train_transforms=False
  test_transforms=False

  if transformation==True:
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

  return (train_data,test_data)

def dataloader(dataset, arguments=False):

  train_data, test_data = dataset
  test_loader = torch.utils.data.DataLoader(test_data, **arguments)
  train_loader = torch.utils.data.DataLoader(train_data, **arguments)

  return train_loader,test_loader

def check_images(train_loader, no_of_images, cmap='gray'):

  batch_data, batch_label = next(iter(train_loader))

  fig = plt.figure()

  for i in range(no_of_images):
    plt.subplot(4, 4, i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap=cmap)
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

def model_summary(input_size):
  device = torch.device("cuda" if check_cuda() else "cpu")
  model_ = model.Net().to(device)
  summary(model_, input_size=input_size)