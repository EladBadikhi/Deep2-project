import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from ResNet import resnet18, ResNet, ResNetBottleNeckBlock
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import recoverability
# params = {'epochs': 20, 'batch_size': 128, 'deepths': [8], 'blocks_sizes': [128]} # 98% train, 75% test
params = {'epochs': 20, 'batch_size': 128, 'deepths': [8], 'blocks_sizes': [128]}#98.852 % train, 76.0% test
if __name__ == '__main__':
    args = sys.argv[1:]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    p = './trained_nets/cifar_net.pth' #Path to trained net
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
    print("before train")
    if not ((len(args) > 0 and args[0] == "NoTrain") or gettrace):
        # from torchsummary import summary
        print(f"{len(args)=}")
        if len(args)>0:
            print(args[0])
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
    #weight pruning
    print("going into weight pruning")
    run_new = True
    if run_new:
        Big_data = 20 #  len(trainloader) = 390
        with torch.no_grad():
            images = [data[0].to(device) for q, data in enumerate(trainloader) if q < Big_data]
            x = torch.cat(images)
            print("layer size: ",net.encoder.gate[0].weight.shape)
            pruned, _ = recoverability.prune_layer(net.encoder.gate[0], x)
            images = [net.encoder.gate[0](data) for data in images]
            net.encoder.gate[0].weight = pruned.weight
            #resnat shape
            #self.encoder.blocks[0].blocks # should be 8 rn
            #self.encoder.blocks[0].blocks[j].blocks # should be about 5 one (conv, batch norm) then one relu alternating 
            #self.encoder.blocks[0].blocks[j].blocks[k].conv.weight

            for j, sequence in enumerate(net.encoder.blocks[0].blocks):
                for k, block in enumerate(sequence.blocks):
                    nimages = [block(data) for data in images]
                    # y = block(x) #changed sisnce push ***
                    if not getattr(block, 'conv', None):
                        images = nimages
                        continue
                    print(f"{j=} {k=}")
                    print("layer size: ",block.conv.weight.shape)
                    x = torch.cat(images)  
                    pruned, _ = recoverability.prune_layer(block.conv, x)
                    images = nimages
                    block.conv.weight = pruned.weight
    
    ############ Test ##########
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(trainloader):
            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the train images: {100 * correct / total} %')
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

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')