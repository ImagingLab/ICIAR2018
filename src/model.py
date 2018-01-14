import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from .options import ModelOptions
from .bach_dataset import PatchDataset, LABELS


class BachModel:
    def __init__(self, model):
        args = ModelOptions().parse()
        weights = args.checkpoints_dir + '/weights_' + model.name() + '.pth'

        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        if os.path.exists(weights):
            print('loading model...\n')
            model = torch.load(weights).cuda()

        self.args = args
        self.weights = weights
        self.model = model.cuda() if args.cuda else model

    def train(self):
        self.model.train()
        print('Start training...\n')

        train_loader = DataLoader(
            dataset=PatchDataset(path=self.args.dataset_path + '/train', stride=self.args.patch_stride, rotate=True, flip=True),
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

                if index > 0 and index % self.args.log_interval == 0:
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
            torch.save(self.model, self.weights)

    def test(self):
        self.model.eval()

        test_loss = 0
        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        test_loader = DataLoader(
            dataset=PatchDataset(path=self.args.dataset_path + '/train', stride=self.args.patch_stride),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=8
        )

        for images, labels in test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            output = self.model(Variable(images, volatile=True))

            test_loss += F.nll_loss(output, Variable(labels), size_average=False).data[0]
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)

            for label in range(classes):
                t_labels = labels == label
                p_labels = predicted == label
                tp[label] += torch.sum(t_labels == (p_labels * 2 - 1))
                tpfp[label] += torch.sum(p_labels)
                tpfn[label] += torch.sum(t_labels)

        for label in range(classes):
            precision[label] += (tp[label] / (tpfp[label] + 1e-8))
            recall[label] += (tp[label] / (tpfn[label] + 1e-8))
            f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

        test_loss /= len(test_loader.dataset)
        print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)
        ))

        for label in range(classes):
            print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                LABELS[label],
                precision[label],
                recall[label],
                f1[label]
            ))

        print('')

