import torch
import torch.nn as nn

from model import GoogLeNet

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda import is_available

device = torch.device('cuda:0' if is_available() else 'cpu')

from config import num_classes, in_channels, num_epoch, batch_size, learning_rate , dev


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(
            module.weight, gain=nn.init.calculate_gain('leaky_relu'))
    elif isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight)

def main():
    # parser = argparse.ArgumentParser(description='PyTorch GoogLeNet Model')
    # parser.add_argument('--lr', default=0.001,
    #                     type=float, help='learning rate')
    # parser.add_argument('-d', '--dev', action="store_true",
    #                     help="To test with single input or complete train")

    # args = parser.parse_args()
    model = GoogLeNet(in_channels=in_channels,
                      num_classes=num_classes).to(device=device)
    model.apply(init_weights)
    model.to(device=device)

    if dev:
        dataset = torchvision.datasets.CIFAR10(
            root='dataset/', train=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))]), download=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(device=device).to(device=device)
        step = 0
        loss_writer = SummaryWriter(f'runs/loss')
        accuracy_writer = SummaryWriter(f'runs/accuracy')

        for epoch in range(num_epoch):
            count = 0
            accuracy_tot = 0
            for _, (data, targets) in enumerate(loader):
                data = data.to(device)
                targets = targets.to(device)
                scores = model(data)
                loss = criterion(scores, targets)

                _, predictions = scores.max(1)
                num_correct = (predictions == targets).sum()
                accuracy = float(num_correct) / float(data.shape[0])

                loss_writer.add_scalar('Training Loss', loss, global_step=step)
                accuracy_writer.add_scalar(
                    'Training Accuracy', accuracy, global_step=step)
                step += 1
                count += 1
                accuracy_tot += accuracy
                optim.zero_grad()
                loss.backward()

                optim.step()
            print("Epoch: {0} Accuracy: {1}".format(
                epoch, float(accuracy_tot)/float(count)))
            count = 0
            accuracy_tot = 0

        # save model
        torch.save(model.state_dict(),
                   "./drive/MyDrive/logs/model.pth")
    else:
        count = 10
        inputs = []
        for i in range(count):
            xin = torch.randn(size=(1, 3, 224, 224))
            inputs.append(xin)

        x = torch.cat(inputs, dim=0)
        output = model(x)

        print(output.shape)