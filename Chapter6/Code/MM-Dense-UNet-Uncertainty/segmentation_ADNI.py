import numpy as np
import torch
import pytu
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from transforms import Transforms, pool_inputs, pool_labels, min_max_normalize, normalize, lab_one_hot
from data_loaders import get_dataset
from splitter import Splitter
from mixed_dense_unet import MixedDenseUNet3D
from strategies import SimpleStrategy
from pytu.iterators import Trainer, Tester
from objectives import MultiTaskLoss
from metrics import get_dice, accuracy, mae
#import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] ="3"


#@pytu.autosacred(name='MixedUNET', dbname='experiments')
#def main(batch_size=4, learning_rate=0.00001, epochs=200, use_multi_gpu=True,
#         load_weights=False,
#         high_order_mode=True,
#         train_model=True,
#         check_train=False,
#         check_valid=False):

batch_size=4
learning_rate=0.00001
epochs=200
use_multi_gpu=True,
load_weights=True,
high_order_mode=True,
train_model=True,
check_train=False,
check_valid=False


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

print('Preparing data...')

#trans = Transforms([min_max_normalize, pool_inputs, pool_labels])

#trans = Transforms([normalize,pool_inputs, pool_labels])

trans = Transforms([min_max_normalize])

DATA_INFO = {'DATA': 'ADNI', 'TRANSFORMS': trans}

trainloader = DataLoader(get_dataset(DATA_INFO, train=True, valid=False, test=False),
                         batch_size=batch_size, shuffle=True)
validloader = DataLoader(get_dataset(DATA_INFO, train=False, valid=True, test=False),
                         batch_size=batch_size, shuffle=False)
testloader = DataLoader(get_dataset(DATA_INFO, train=False, valid=False, test=True),
                        batch_size=batch_size, shuffle=False)

print('Building network...')

print("Using a Dense Model")
denseunet = MixedDenseUNet3D(
    in_conv=1,
    out_conv_seg=8,
    out_conv_class=2,
    out_conv_reg=1,
    in_meta=52,
    out_meta=0,
    growth_rate=1,
    activation=torch.nn.functional.leaky_relu,
    n_layer=4,
    n_pool=4,
    pool_type='max',
    padding_type='same_replicate',
    init_behaviour='normalized',
    high_order_mode=high_order_mode,
    kernel_size=3)

network = Splitter(denseunet)


tissue_weights = np.load('../Data_Info/Tissue_Weights_sub.npy').astype('float32')
tissue_weights = tissue_weights[:-1] 
tissue_weights = tissue_weights/tissue_weights.sum()
print("using Tissue Weights: ")


#Current_Dice_Scores = np.array([0.982,0.921,0.798,0.727,0.763,0.698,0.683,0.633,0.525]).astype('float32')
#Current_Dice_Scores = 1/Current_Dice_Scores
#Current_Dice_Scores = Current_Dice_Scores/Current_Dice_Scores.sum()

#tissue_weights = tissue_weights * Current_Dice_Scores
#tissue_weights = tissue_weights/tissue_weights.sum()

print(tissue_weights)
tissue_weights = torch.from_numpy(tissue_weights).cuda() 


class_weights = np.array([0.5, 0.5]).astype('float32')
print("using Class Weights: ")
print(class_weights)
class_weights = torch.from_numpy(class_weights).cuda()


strat = SimpleStrategy(network, optimizer=Adam(network.parameters(), lr=learning_rate),
                       loss=MultiTaskLoss(tissue_weights=tissue_weights, class_weights=class_weights,
                                          use_uncertainity_multitask=True, use_geometrical_multitask=False,
                                          multitask_weight=[1.0, 1.0, 1.0], reg_criterion=torch.nn.SmoothL1Loss()),
                       multi_gpu=use_multi_gpu)

if load_weights:
  strat.load_weights('./../Mixed_Dense_U_NET_Multi_Seg/model_out_0_32/best_validation_loss.pth')

strat.read_data = lab_one_hot

file = open("../Data_Info/Names_sub_noCC.txt", "r")
labels_names = []
for line in file:
    labels_names.append(line.strip())

if train_model:
    print('Starting training...')
    trainer = Trainer(trainloader,
                      validloader,
                      strat,
                      tensorboard=True,
                      save_criterions='validation_loss')

    for i in range(len(labels_names)):
        trainer.add_metric(get_dice(i, labels_names[i]))

    trainer.add_metric(accuracy)
    trainer.add_metric(mae)
    trainer.train(epochs, freq=1)

# After training

strat.load_weights('model_out/best_validation_loss.pth')

if check_train:
    tester_Train = Tester(trainloader, strat)

    for i in range(len(labels_names)):
        tester_Train.add_metric(get_dice(i, labels_names[i]))

    tester_Train.add_metric(accuracy)
    tester_Train.add_metric(mae)
    tester_Train.test()

if check_valid:
    print('Starting testing (Validation)...')

    tester_Validation = Tester(validloader, strat)

    for i in range(len(labels_names)):
        tester_Validation.add_metric(get_dice(i, labels_names[i]))

    tester_Validation.add_metric(accuracy)
    tester_Validation.add_metric(mae)
    tester_Validation.test()

print('Starting testing (Testing)...')

tester_Testing = Tester(testloader, strat)

for i in range(len(labels_names)):
    tester_Testing.add_metric(get_dice(i, labels_names[i]))

tester_Testing.add_metric(accuracy)
tester_Testing.add_metric(mae)
tester_Testing.test()
