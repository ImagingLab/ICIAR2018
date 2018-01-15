import os
import time
import datetime
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from .datasets import *

TRAIN_PATH = '/train'
TEST_PATH = '/test'


class PatchWiseModel:
    def __init__(self, args, network):
        weights = args.checkpoints_dir + '/weights_' + network.name() + '.pth'

        if os.path.exists(weights):
            print('Loading "patch-wise" model...')
            network = torch.load(weights).cuda()

        self.args = args
        self.weights = weights
        self.network = network.cuda() if args.cuda else network

    def start_train(self):
        self.network.train()
        print('Start training: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        train_loader = DataLoader(
            dataset=PatchWiseDataset(path=self.args.dataset_path + TRAIN_PATH, stride=self.args.patch_stride, rotate=True, flip=True),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epc: 2 ** (epc // 10))
        best = 0
        mean = 0
        
        for epoch in range(1, self.args.epochs + 1):

            self.network.train()
            scheduler.step()
            stime = datetime.datetime.now()

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(train_loader):

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
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

            print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))
            acc = self.start_test()
            mean += acc
            if acc > best:
                best = acc
                print('Saving model to "{}"'.format(self.weights))
                torch.save(self.network, self.weights)

                print('\nEnd of training, best accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def start_test(self):
        self.network.eval()

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
            dataset=PatchWiseDataset(path=self.args.dataset_path + TEST_PATH, stride=self.args.patch_stride),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        print('\nEvaluating....')
        for images, labels in test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            output = self.network(Variable(images, volatile=True))

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
        acc = 100. * correct / len(test_loader.dataset)
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
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
        return acc

    def output_train(self):
        return self._output(True)

    def output_test(self):
        return self._output(False)

    def _output(self, is_train):
        self.network.eval()

        dataset = ImageWiseDataset(
            path=self.args.dataset_path + (TRAIN_PATH if is_train else TEST_PATH),
            stride=self.args.patch_stride,
            flip=(True if is_train else False))

        output_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8)
        output_images = []
        output_labels = []

        for index, (images, labels) in enumerate(output_loader):
            if index > 0 and index % 100 == 0:
                print('{} images loaded'.format(index))

            if self.args.cuda:
                images = images.cuda()
                res = self.network.features(Variable(images[0], volatile=True))
                output_labels.append(labels.numpy())
                output_images.append(res.squeeze().data.cpu().numpy())

        return torch.from_numpy(np.array(output_images)), torch.from_numpy(np.array(output_labels)).squeeze()


class ImageWiseModel:
    def __init__(self, args, image_wise_network, patch_wise_network):
        weights = args.checkpoints_dir + '/weights_' + image_wise_network.name() + '.pth'

        if os.path.exists(weights):
            print('\nLoading "image-wise" model...')
            image_wise_network = torch.load(weights).cuda()

        self.args = args
        self.weights = weights
        self.patch_wise_model = PatchWiseModel(args, patch_wise_network)
        self.network = image_wise_network.cuda() if args.cuda else image_wise_network
        self._test_loader = None

    def start_train(self):
        self.network.train()
        print('Evaluating patch-wise model...')

        patch_outputs = self.patch_wise_model.output_train()
        train_loader = DataLoader(
            dataset=TensorDataset(patch_outputs[0], patch_outputs[1]),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )

        print('Start training: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epc: 2 ** (epc // 10))
        best = 0
        mean = 0

        for epoch in range(1, self.args.epochs + 1):

            self.network.train()
            scheduler.step()
            stime = datetime.datetime.now()

            correct = 0
            total = 0

            for index, (images, labels) in enumerate(train_loader):

                if self.args.cuda:
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                output = self.network(Variable(images))
                loss = F.nll_loss(output, Variable(labels))
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(output.data, 1)
                correct += torch.sum(predicted == labels)
                total += len(images)

                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                    epoch,
                    index * len(images),
                    len(train_loader.dataset),
                    100. * index / len(train_loader),
                    loss.data[0],
                    100 * correct / total
                ))

            print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))
            acc = self.start_test()
            mean += acc
            if acc > best:
                best = acc
                print('Saving model to "{}"'.format(self.weights))
                torch.save(self.network, self.weights)

        print('\nEnd of training, best accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def start_test(self):
        self.network.eval()

        if self._test_loader is None:
            patch_outputs = self.patch_wise_model.output_test()
            self._test_loader = DataLoader(
                dataset=TensorDataset(patch_outputs[0], patch_outputs[1]),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=2
            )

        test_loss = 0
        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        print('\nEvaluating....')
        for images, labels in self._test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            output = self.network(Variable(images, volatile=True))

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

        test_loss /= len(self._test_loader.dataset)
        acc = 100. * correct / len(self._test_loader.dataset)
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss,
            correct,
            len(self._test_loader.dataset),
            acc
        ))

        for label in range(classes):
            print('{}:  \t Precision: {:.2f},  Recall: {:.2f},  F1: {:.2f}'.format(
                LABELS[label],
                precision[label],
                recall[label],
                f1[label]
            ))

        print('')
        return acc
