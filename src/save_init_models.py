from cnn import CNN
from fcn import FCN
from svm import SVM
import torch

INIT_CNN = False
INIT_FCN = False
INIT_SVM = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if INIT_CNN:
    model = CNN().to(device)
    init_path = '../init/mnist_cnn.init'
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))

if INIT_FCN:
    model = FCN().to(device)
    init_path = '../init/mnist_fcn.init'
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))

if INIT_SVM:
    model = SVM()
    init_path = '../init/mnist_svm.init'
    torch.save(model.state_dict(), init_path)
    print('Save init: {}'.format(init_path))
