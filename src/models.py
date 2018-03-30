import time
import time
import ntpath
import datetime
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as ply
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from .datasets import *

TRAIN_PATH = '/train'
VALIDATION_PATH = '/validation'


class BaseModel:
    def __init__(self, args, network, weights_path):
        self.args = args
        self.weights = weights_path
        self.network = network.cuda() if args.cuda else network
        self.load()

    def load(self):
        try:
            if os.path.exists(self.weights):
                print('Loading "patch-wise" model...')
                self.network.load_state_dict(torch.load(self.weights))
        except:
            print('Failed to load pre-trained network')

    def save(self):
        print('Saving model to "{}"'.format(self.weights))
        torch.save(self.network.state_dict(), self.weights)


class PatchWiseModel(BaseModel):
    def __init__(self, args, network):
        super(PatchWiseModel, self).__init__(args, network, args.checkpoints_path + '/weights_' + network.name() + '.pth')

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

            self.save()

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

    def test(self, path, verbose=True):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE, augment=False)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        stime = datetime.datetime.now()

        if verbose:
            print('\t sum\t\t max\t\t maj')

        res = []

        for index, (image, file_name) in enumerate(data_loader):
            image = image.squeeze()
            if self.args.cuda:
                image = image.cuda()

            output = self.network(Variable(image))
            _, predicted = torch.max(output.data, 1)

            #
            # the following measures are prioritised based on [invasive, insitu, benign, normal]
            # the original labels are [normal, benign, insitu, invasive], so we reverse the order using [::-1]
            # output data shape is 12x4
            # sum_prop: sum of probabilities among y axis: (1, 4), reverse, and take the index  of the largest value
            # max_prop: max of probabilities among y axis: (1, 4), reverse, and take the index  of the largest value
            # maj_prop: majority voting: create a one-hot vector of predicted values: (12, 4), sum among y axis: (1, 4), reverse, and take the index  of the largest value

            sum_prob = 3 - np.argmax(np.sum(np.exp(output.data.cpu().numpy()), axis=0)[::-1])
            max_prob = 3 - np.argmax(np.max(np.exp(output.data.cpu().numpy()), axis=0)[::-1])
            maj_prob = 3 - np.argmax(np.sum(np.eye(4)[np.array(predicted).reshape(-1)], axis=0)[::-1])

            res.append([sum_prob, max_prob, maj_prob, file_name[0]])

            if verbose:
                np.sum(output.data.cpu().numpy(), axis=0)
                print('{}) \t {} \t {} \t {} \t {}'.format(
                    str(index + 1).rjust(2, '0'),
                    LABELS[sum_prob].ljust(8),
                    LABELS[max_prob].ljust(8),
                    LABELS[maj_prob].ljust(8),
                    ntpath.basename(file_name[0])))

        if verbose:
            print('\nInference time: {}\n'.format(datetime.datetime.now() - stime))

        return res

    def output(self, input_tensor):
        self.network.eval()
        res = self.network.features(Variable(input_tensor, volatile=True))
        return res.squeeze()

    def visualize(self, path, channel=0):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        for index, (image, file_name) in enumerate(data_loader):

            if self.args.cuda:
                image = image[0].cuda()

            patches = self.output(image)

            output = patches.cpu().data.numpy()

            map = np.zeros((3 * 64, 4 * 64))

            for i in range(12):
                row = i // 4
                col = i % 4
                map[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = output[i]

            if len(map.shape) > 2:
                map = map[channel]

            with Image.open(file_name[0]) as img:
                ply.subplot(121)
                ply.axis('off')
                ply.imshow(np.array(img))

                ply.subplot(122)
                ply.imshow(map, cmap='gray')
                ply.axis('off')

                ply.show()


class ImageWiseModel(BaseModel):
    def __init__(self, args, image_wise_network, patch_wise_network):
        super(ImageWiseModel, self).__init__(args, image_wise_network, args.checkpoints_path + '/weights_' + image_wise_network.name() + '.pth')

        self.patch_wise_model = PatchWiseModel(args, patch_wise_network)
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
                self.save()

        print('\nEnd of training, best accuracy: {}, mean accuracy: {}\n'.format(best, mean // epoch))

    def validate(self, verbose=True, roc=False):
        self.network.eval()

        if self._test_loader is None:
            self._test_loader = self._patch_loader(self.args.dataset_path + VALIDATION_PATH, False)

        val_loss = 0
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

            val_loss += F.nll_loss(output, Variable(labels), size_average=False).data[0]
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

        val_loss /= len(self._test_loader.dataset)
        acc = 100. * correct / len(self._test_loader.dataset)

        if roc == 1:
            labels_true = label_binarize(labels_true, classes=range(classes))
            for lbl in range(classes):
                fpr, tpr, _ = roc_curve(labels_true[:, lbl], labels_pred[:, lbl])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label='{} (AUC: {:.1f})'.format(LABELS[lbl], roc_auc * 100))

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
                val_loss,
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

    def test(self, path, verbose=True, ensemble=True):
        self.network.eval()
        dataset = TestDataset(path=path, stride=PATCH_SIZE, augment=ensemble)
        data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
        stime = datetime.datetime.now()

        if verbose:
            print('')

        res = []

        for index, (image, file_name) in enumerate(data_loader):
            n_bins, n_patches = image.shape[1], image.shape[2]
            image = image.view(-1, 3, PATCH_SIZE, PATCH_SIZE)

            if self.args.cuda:
                image = image.cuda()

            patches = self.patch_wise_model.output(image)
            patches = patches.view(n_bins, -1, 64, 64)

            if self.args.cuda:
                patches = patches.cuda()

            output = self.network(patches)
            _, predicted = torch.max(output.data, 1)

            # maj_prop: majority voting: create a one-hot vector of predicted values: (12, 4),
            # sum among y axis: (1, 4), reverse, and take the index  of the largest value

            maj_prob = 3 - np.argmax(np.sum(np.eye(4)[np.array(predicted).reshape(-1)], axis=0)[::-1])

            confidence = np.sum(np.array(predicted) == maj_prob) / n_bins if ensemble else torch.max(torch.exp(output.data))
            confidence = np.round(confidence * 100, 2)

            res.append([maj_prob, confidence, file_name[0]])

            if verbose:
                print('{}) {} ({}%) \t {}'.format(
                    str(index).rjust(2, '0'),
                    LABELS[maj_prob],
                    confidence,
                    ntpath.basename(file_name[0])))

        if verbose:
            print('\nInference time: {}\n'.format(datetime.datetime.now() - stime))

        return res

    def _patch_loader(self, path, augment):
        images_path = '{}/{}_images.npy'.format(self.args.checkpoints_path, self.network.name())
        labels_path = '{}/{}_labels.npy'.format(self.args.checkpoints_path, self.network.name())

        if self.args.debug and augment and os.path.exists(images_path):
            np_images = np.load(images_path)
            np_labels = np.load(labels_path)

        else:
            dataset = ImageWiseDataset(
                path=path,
                stride=PATCH_SIZE,
                flip=augment,
                rotate=augment,
                enhance=augment)

            bsize = 8
            output_loader = DataLoader(dataset=dataset, batch_size=bsize, shuffle=True, num_workers=4)
            output_images = []
            output_labels = []

            for index, (images, labels) in enumerate(output_loader):
                if index > 0 and index % 10 == 0:
                    print('{} images loaded'.format(int((index * bsize) / dataset.augment_size)))

                if self.args.cuda:
                    images = images.cuda()

                bsize = images.shape[0]

                res = self.patch_wise_model.output(images.view((-1, 3, 512, 512)))
                res = res.view((bsize, -1, 64, 64)).data.cpu().numpy()

                for i in range(bsize):
                    output_images.append(res[i])
                    output_labels.append(labels.numpy()[i])

            np_images = np.array(output_images)
            np_labels = np.array(output_labels)

            if self.args.debug and augment:
                np.save(images_path, np_images)
                np.save(labels_path, np_labels)

        images, labels = torch.from_numpy(np_images), torch.from_numpy(np_labels).squeeze()

        return DataLoader(
            dataset=TensorDataset(images, labels),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=2
        )
