import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from ResNet import resnet18, ResNet, ResNetBottleNeckBlock
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
# params = {'epochs': 20, 'batch_size': 128, 'deepths': [8], 'blocks_sizes': [128]} # 98% train, 75% test
params = {'epochs': 20, 'batch_size': 128, 'deepths': [8], 'blocks_sizes': [128]}
if __name__ == '__main__':
    args = sys.argv[1:]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = './trained_nets/cifar_net.pth'
    print(f'{device=}')
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = params['batch_size']

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = ResNet(3, 10, block=ResNetBottleNeckBlock, deepths=params['deepths'], blocks_sizes=params['blocks_sizes'])
    net.to(device)
    gettrace = getattr(sys, 'gettrace', None)
    if not (len(args) > 0 and args[0] == "NoTrain" or gettrace):
        # from torchsummary import summary
        print(f"{len(args)=}")
        if len(args)>0:
            print(args[0])
        # model = resnet18(3, 1000)
        # summary(net.cuda(), (3, 224, 224))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        print('Starting training')
        for epoch in range(params['epochs']):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        print('Finished Training')
        torch.save(net.state_dict(), p)
        if (Path(p).exists()):
            print("Saved model")
    else:
        net.load_state_dict(torch.load(p, map_location=torch.device(device)))
    ############ Pruning ##########
    with torch.no_grad():
        # print(f"{net.encoder.gate[0].weight[:,2:,3,4]=}")
        # net.encoder.gate[0].weight[:,2:,3,4]=0
        print(net.decoder.decoder.weight)
        # net.decoder.decoder.weight.data = torch.zeros_like(net.decoder.decoder.weight)
        # print(f"{net.encoder.gate[0].weight[:,2:,3,4]=}")
        # print(f"{type(net.encoder.blocks[0])=}, {vars(net.encoder.blocks[0])=}") # Go over all of these! in the depth thar is need to get to all the layers
    ############ Test ##########
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in trainloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the train images: {100 * correct // total} %')
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')