from model.ann_model.U_Net import *
from snn import simulation
import os
from snn.conversion import Parser
import torch
from torch import optim

timesteps = 30
dataset_name = "ISBI_2012"
base_path = './test/seg_train'
method = "connection_wise"
scale_method = "robust"
neuron_class = "multi"
reset_method = 'reset_by_subtraction'
vth = 1.0
opts = "adam"
batch_size = 16
learning_rate = 1e-6
epochs = 100
seed1 = "42"
seed2 = "52"
seed3 = "21"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kwargs = { }
str_n_join = ['optim', opts, 'batch_size', batch_size, 'lr', learning_rate, 'ep', epochs, 'm', method, 't', timesteps, 'neuron', neuron_class]
str_s_join = ['s1', seed1, 's2', seed2, 's3', seed3] + str_n_join
str_n = '_'.join(str(s) for s in str_n_join)
str_s = '_'.join(str(s) for s in str_s_join)

post_path = os.path.join(dataset_name, scale_method, reset_method, str_n)
path = os.path.join(base_path, post_path)
post_log_path = os.path.join('logs', dataset_name, scale_method, reset_method, str_s)
logs_path  = os.path.join(base_path, post_log_path)

parser = Parser(path = './lambda_factor/ISBI_2012')
pytorch_model = Segmentation_UNet(input_channel=3, class_num=2, fnum=64) 
random_tensor = torch.randn((batch_size, 3, 32, 32), dtype= torch.float32)

temp = pytorch_model(x = random_tensor, input_type = "original")
print("segment mask shape ", temp.shape)
parser_model = parser.parse(pytorch_model,random_tensor, method=method, scale_method=scale_method)

snn_model = parser.convert_to_snn(parser_model, neuron_class=neuron_class, timesteps=timesteps, reset_method=reset_method, v_threshold=vth)
snn_model.to(device)

model_path     = os.path.join(path, 'snn_model', 'snn_model.pth')


simulator = simulation.simulator(timesteps=timesteps, dataset_name=dataset_name, 
                                    path=path, logs_path=logs_path, device=device, **kwargs)

train_data = random_tensor.to(device=device, dtype=torch.float32)
train_label = random_tensor.to(device=device, dtype=torch.long)
train_label = torch.squeeze(train_label, dim=1)

output = simulator.simulate_for_sample(snn_model, train_data)
print("segment mask shape ", output.shape)