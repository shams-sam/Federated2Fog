from arguments import Arguments
from fcn import FCN
from svm import SVM
import torch

INIT_FCN = True
INIT_SVM = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Arguments()

for dataset in args.input_sizes:
    if INIT_FCN:
        model = FCN(args.input_sizes[dataset],
                    args.output_sizes[dataset]).to(device)
        print(model.input_size, model.output_size)
        init_path = '../init/{}_fcn.init'.format(dataset)
        torch.save(model.state_dict(), init_path)
        print('Save init: {}'.format(init_path))

    if INIT_SVM:
        model = SVM(args.input_sizes[dataset],
                    args.output_sizes[dataset]).to(device)
        print(model.n_feature, model.n_class)
        init_path = '../init/{}_svm.init'.format(dataset)
        torch.save(model.state_dict(), init_path)
        print('Save init: {}'.format(init_path))
