import torchvision
from torch.utils.data import DataLoader
from model import GoogLeNet
import torch


def get_model():
    model = GoogLeNet(in_channels=3, num_classes=10)
    model.load_state_dict(torch.load('./model.pth'))
    model.eval()

    return model


def test():
    dataset = torchvision.datasets.CIFAR10(root='./dataset/', train=True,
                                           transform=torchvision.transforms.ToTensor(), download=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = get_model()
    model.train(False)  # set model to inference mode

    for i, (data, target) in enumerate(loader):
        if i == 10:
            break
        output = model(data)
        _, predictions = output.max(1)

        for i, pred in enumerate(predictions):
            print(pred.item(), target[i].item())
        
        print()


if __name__ == '__main__':
    test()
