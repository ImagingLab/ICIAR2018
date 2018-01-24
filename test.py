from src import *

args = ModelOptions().parse()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pw_network = PatchWiseNetwork(args.channels)
iw_network = ImageWiseNetwork(args.channels)

if args.testset_path is '':
    import tkinter.filedialog as fdialog

    args.testset_path = fdialog.askopenfilename(initialdir=r"./dataset/test", title="choose your file", filetypes=(("tiff files", "*.tif"), ("all files", "*.*")))

if args.network == '1':
    pw_model = PatchWiseModel(args, pw_network)
    pw_model.test(args.testset_path)

else:
    im_model = ImageWiseModel(args, iw_network, pw_network)
    im_model.test(args.testset_path, ensemble=args.ensemble == 1)
