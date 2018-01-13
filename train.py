from src import ModelOptions, BachModel, BachNetwork, BachNetwork2, AlexNet

opt = ModelOptions().parse()

if opt.network == 'AlexNet':
    network = AlexNet()

elif opt.network == 'BachNetwork2':
    network = BachNetwork2()

else:
    network = BachNetwork()

model = BachModel(network)
model.train()
#model.test()
