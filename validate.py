from src import *

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pw_network = PatchWiseNetwork(args.channels)
iw_network = ImageWiseNetwork(args.channels)

iw_model = ImageWiseModel(args, iw_network, pw_network)
iw_model.validate(roc=True)
