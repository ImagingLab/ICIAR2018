from src import *

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pw_network = PatchWiseNetwork1()
iw_network = ImageWiseNetwork2()

#pw_model = PatchWiseModel(args, pw_network)
#pw_model.train()

iw_model = ImageWiseModel(args, iw_network, pw_network)
iw_model.train()
