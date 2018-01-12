from src import BachModel

model = BachModel()
model.train()

# import torch
# from torchvision import datasets, transforms
#
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=64, shuffle=True)
#
#
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(target[10])
#     break
