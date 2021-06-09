import torch
import torchvision.models as models
from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
from robustness.datasets import CINIC
from robustness.model_utils import make_and_restore_model
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read dataset from file
dataset = CIFAR()

# read trained model
model_rn_0, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=dataset, resume_path='cifar_nat_0.pt')
model_rn_025, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=dataset, resume_path='cifar_l2_0_25.pt')
model_rn_05, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=dataset, resume_path='cifar_l2_0_5.pt')
model_rn_1, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=dataset, resume_path='cifar_l2_1_0.pt')


# img = img.to(device)
batch_size = 10
element_size = 3*32*32

inp_data = np.zeros((1,3*32*32))
op_data = np.zeros((1,10))
label_data = np.zeros((1,1))
acc_data = np.array([])

count_beat = 0

# create dataset loader
train_loader, _ = dataset.make_loaders(workers=0, batch_size=10)


for img, label in train_loader:
    img = img.to(device)
    op = model_rn_0(img)
    op_data = np.vstack((op_data, op[0].detach().cpu().numpy()))
    inp_data = np.vstack((inp_data, img.cpu().numpy().reshape(batch_size, element_size)))
    label_data = np.append(label_data, label.cpu().numpy())
    count_beat = count_beat + 1
    print(count_beat)

    # if count_beat == 1: break;

inp_data = inp_data[1:,:]
op_data = op_data[1:,:]
label_data = label_data[1:]
acc_data = np.append(acc_data, np.sum(np.argmax(op_data,1) == label_data)/op_data.shape[0])

np.savetxt('train_eval_results/testset_input_0.csv', inp_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_output_0.csv', op_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_label_0.csv', label_data, fmt="%d")


# 0.25 =========================
inp_data = np.zeros((1,3*32*32))
op_data = np.zeros((1,10))
label_data = np.zeros((1,1))
train_loader, _ = dataset.make_loaders(workers=0, batch_size=10)

count_beat = 0

for img, label in train_loader:
    img = img.to(device)
    # target_label = (label + torch.randint_like(label, high=3))%10
    op = model_rn_025(img)
    # adv_op = model_rn_025(img, target_label, **attack_kwargs)
    inp_data = np.vstack((inp_data, img.cpu().numpy().reshape(batch_size, element_size)))
    op_data = np.vstack((op_data, op[0].detach().cpu().numpy()))
    # op_data = np.vstack((op_data, adv_op[0].detach().cpu().numpy()))
    label_data = np.append(label_data, label.cpu().numpy())
    count_beat = count_beat + 1
    print(count_beat)

inp_data = inp_data[1:,:]
op_data = op_data[1:,:]
label_data = label_data[1:]
acc_data = np.append(acc_data, np.sum(np.argmax(op_data,1) == label_data)/op_data.shape[0])
np.savetxt('train_eval_results/testset_input_025.csv', inp_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_output_025.csv', op_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_label_025.csv', label_data, fmt="%d")

# # 0.5 =================================
inp_data = np.zeros((1,3*32*32))
op_data = np.zeros((1,10))
label_data = np.zeros((1,1))
train_loader, _ = dataset.make_loaders(workers=0, batch_size=10)

count_beat = 0

for img, label in train_loader:
    img = img.to(device)
    op = model_rn_05(img)
    op_data = np.vstack((op_data, op[0].detach().cpu().numpy()))
    inp_data = np.vstack((inp_data, img.cpu().numpy().reshape(batch_size, element_size)))
    label_data = np.append(label_data, label.cpu().numpy())
    count_beat = count_beat + 1
    print(count_beat)

inp_data = inp_data[1:,:]
op_data = op_data[1:,:]
label_data = label_data[1:]
acc_data = np.append(acc_data, np.sum(np.argmax(op_data,1) == label_data)/op_data.shape[0])
np.savetxt('train_eval_results/testset_input_05.csv', inp_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_output_05.csv', op_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_label_05.csv', label_data, fmt="%d")

# # # 1.0 =================================
inp_data = np.zeros((1,3*32*32))
op_data = np.zeros((1,10))
label_data = np.zeros((1,1))
train_loader, _ = dataset.make_loaders(workers=0, batch_size=10)

count_beat = 0

for img, label in train_loader:
    img = img.to(device)
    op = model_rn_1(img)
    op_data = np.vstack((op_data, op[0].detach().cpu().numpy()))
    inp_data = np.vstack((inp_data, img.cpu().numpy().reshape(batch_size, element_size)))
    label_data = np.append(label_data, label.cpu().numpy())
    count_beat = count_beat + 1
    print(count_beat)

inp_data = inp_data[1:,:]
op_data = op_data[1:,:]
label_data = label_data[1:]
acc_data = np.append(acc_data,np.sum(np.argmax(op_data,1) == label_data)/op_data.shape[0])
np.savetxt('train_eval_results/testset_input_1.csv', inp_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_output_1.csv', op_data, fmt="%.4f")
np.savetxt('train_eval_results/testset_label_1.csv', label_data, fmt="%d")
