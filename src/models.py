import time
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from .datasets import *

TRAIN_PATH = '/train'
VALIDATION_PATH = '/validation'


class PatchWiseModel:
    def __init__(self, args, network):
        weights = args.checkpoints_dir + '/weights_' + network.name() + '.pth'

        if os.path.exists(weights):
            print('Loading "patch-wise" model...')
            network = torch.load(weights).cuda()

        self.args = args
        self.weights = weights
        self.network = network.cuda() if args.cuda else network

    def train(self):
        self.network.train()
        print('Start training patch-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        train_loader = DataLoader(
            dataset=PatchWiseDataset(path=self.args.dataset_path + TRAIN_PATH, stride=self.args.patch_stride, rotate=True, flip=True, enhance=True),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4
        )
        optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        best = self.validate(verbose=False)
        mean = 0
        epoch = 0

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
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        self.args.epochs,
                        index * len(images),
                        len(train_loader.dataset),
                        100. * index / len(train_loader),
                        loss.data[0],
                        100 * correct / total
                    ))

            print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))
            acc = self.validate()
            mean += acc
            if acc > best:
                best = acc

            print('Saving model to "{}"'.format(self.weights))
            torch.save(self.network, self.weights)

        self.network = torch.load(self.weights).cuda()
        print('\nEnd of training, best accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def validate(self, verbose=True):
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
            dataset=PatchWiseDataset(path=self.args.dataset_path + VALIDATION_PATH, stride=self.args.patch_stride),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )
        if verbose:
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

        if verbose:
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

    def output(self, input_tensor):
        self.network.eval()
        res = self.network.features(Variable(input_tensor, volatile=True))
        return res.squeeze()


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

    def train(self):
        self.network.train()
        print('Evaluating patch-wise model...')

        train_loader = self._patch_loader(self.args.dataset_path + TRAIN_PATH, True)

        print('Start training image-wise network: {}\n'.format(time.strftime('%Y/%m/%d %H:%M')))

        optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        best = self.validate(verbose=False)
        mean = 0
        epoch = 0

        for epoch in range(1, self.args.epochs + 1):

            self.network.train()
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

                if index > 0 and index % 10 == 0:
                    print('Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                        epoch,
                        self.args.epochs,
                        index * len(images),
                        len(train_loader.dataset),
                        100. * index / len(train_loader),
                        loss.data[0],
                        100 * correct / total
                    ))

            print('\nEnd of epoch {}, time: {}'.format(epoch, datetime.datetime.now() - stime))
            acc = self.validate()
            mean += acc
            if acc > best:
                best = acc

            print('Saving model to "{}"'.format(self.weights))
            torch.save(self.network, self.weights)

        self.network = torch.load(self.weights).cuda()
        print('\nEnd of training, best accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def validate(self, verbose=True, roc=False):
        self.network.eval()

        if self._test_loader is None:
            self._test_loader = self._patch_loader(self.args.dataset_path + VALIDATION_PATH, False)

        test_loss = 0
        correct = 0
        classes = len(LABELS)

        tp = [0] * classes
        tpfp = [0] * classes
        tpfn = [0] * classes
        precision = [0] * classes
        recall = [0] * classes
        f1 = [0] * classes

        if verbose:
            print('\nEvaluating....')

        labels_true = []
        labels_pred = np.empty((0, 4))

        for images, labels in self._test_loader:

            if self.args.cuda:
                images, labels = images.cuda(), labels.cuda()

            output = self.network(Variable(images, volatile=True))

            test_loss += F.nll_loss(output, Variable(labels), size_average=False).data[0]
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(predicted == labels)

            labels_true = np.append(labels_true, labels)
            labels_pred = np.append(labels_pred, torch.exp(output.data).cpu().numpy(), axis=0)

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

        if roc == 1:
            labels_true = label_binarize(labels_true, classes=range(classes))
            for lbl in range(classes):
                fpr, tpr, _ = roc_curve(labels_true[:, lbl], labels_pred[:, lbl])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label='{} (area: {:.2f})'.format(LABELS[lbl], roc_auc))

            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc="lower right")
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title('Receiver Operating Characteristic')
            plt.show()

        if verbose:
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

    def test(self, path):
        dataset = TestDataset(path=path, stride=PATCH_SIZE)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        stime = datetime.datetime.now()
        print('')

        for index, (image, file_name) in enumerate(data_loader):

            if self.args.cuda:
                image = image[0].cuda()

            patches = self.patch_wise_model.output(image).unsqueeze(0)

            if self.args.cuda:
                patches = patches.cuda()

            output = self.network(patches)
            _, predicted = torch.max(output.data, 1)
            confidence = np.round(torch.max(torch.exp(output.data)) * 100, 2)

            print('{}) {} ({}%) - {}'.format(index + 1, LABELS[predicted[0]], confidence, file_name[0]))

        print('\nInference time: {}\n'.format(datetime.datetime.now() - stime))

    def _patch_loader(self, path, augment):
        if augment and os.path.exists('np_images.npy'):
            np_images = np.load('np_images.npy')
            np_labels = np.load('np_labels.npy')

        else:
            dataset = ImageWiseDataset(
                path=path,
                stride=PATCH_SIZE,
                flip=augment,
                enhance=augment)

            output_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
            output_images = []
            output_labels = []

            for index, (images, labels) in enumerate(output_loader):
                if index > 0 and index % 100 == 0:
                    print('{} images loaded'.format(int(index / 4)))

                if self.args.cuda:
                    images = images.cuda()

                res = self.patch_wise_model.output(images[0])
                output_labels.append(labels.numpy())
                output_images.append(res.squeeze().data.cpu().numpy())

            np_images = np.array(output_images)
            np_labels = np.array(output_labels)
            if augment:
                np.save('np_images', np_images)
                np.save('np_labels', np_labels)

        images, labels = torch.from_numpy(np_images), torch.from_numpy(np_labels).squeeze()

        return DataLoader(
            dataset=TensorDataset(images, labels),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )
