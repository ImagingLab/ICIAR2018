import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .network import BachNetwork
from .options import ModelOptions
from .bach_dataset import BachDataset


class BachModel:
    def __init__(self):
        model = BachNetwork()
        args = ModelOptions().parse()
        weights = args.checkpoints_dir + '/weights.pth'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

            if os.path.exists(weights):
                model = torch.load(weights).cuda()

        self.args = args
        self.model = model.cuda() if args.cuda else model

    def train(self):
        self.model.train()
        print('Start training...\n')

        train_loader = DataLoader(
            dataset=BachDataset(path=self.args.dataset_path + '/train', stride=self.args.patch_stride, augment=True),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

        for epoch in range(1, self.args.epochs + 1):

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(train_loader):

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.model(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                if index > 0 and index % 20 == 0:
                    print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        index * len(images),
                        len(train_loader.dataset),
                        100. * index / len(train_loader),
                        loss.data[0],
                        100 * correct / total
                    ))

            print('End of epoch {}, saving model....\n'.format(epoch))
            self.test()
            torch.save(self.model, self.args.checkpoints_dir + '/weights.pth')

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0
        test_loader = DataLoader(
            dataset=BachDataset(path=self.args.dataset_path + '/test', stride=self.args.patch_stride, augment=False),
            batch_size=self.args.batch_size,
            shuffle=False
        )

        for images, labels in test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            images, labels = Variable(images, volatile=True), Variable(labels)

            output = self.model(images)

            test_loss += F.nll_loss(output, labels, size_average=False).data[0]  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
