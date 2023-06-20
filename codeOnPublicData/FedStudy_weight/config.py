import argparse
import json


class Config:
    def get_param(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--config', type=str, default='./config/DenseNet.json', help='config path')
        # parser.add_argument('--config', type=str, default='./config/UNet3D.json', help='config path')

        args = parser.parse_args()
        config_path = args.config
        config_param = open(config_path, "r")
        param_str = config_param.read()
        param = json.loads(param_str)

        return param


config = Config()
param = config.get_param()

task = param["task"]

model = param["model"]

data_param = param["data_param"]
data_random_split = data_param["data_random_split"]
data_path = data_param["data"]
data_type = data_param["data_type"]
channel = data_param["channel"]
num_classes = data_param["num_classes"]
img_size = data_param["img_size"]
rotate = data_param["rotate"]

bench_param = param["bench_param"]
server_address = bench_param["server_address"]
DEVICE = bench_param["device"]
num_rounds = bench_param["num_rounds"]

training_param = param["training_param"]
epochs = training_param["epochs"]
batch_size = training_param["batch_size"]
lr = training_param["learning_rate"]
loss_func = training_param["loss_func"]
optimiz = training_param["optimizer"]
optimizer_param = training_param["optimizer_param"]

if optimiz == "sgd" or optimiz == "FedSgd":
    momentum = optimizer_param["momentum"]
    dampening = optimizer_param["dampening"]
    weight_decay = optimizer_param['weight_decay']
    nesterov = optimizer_param['nesterov']

elif optimiz == "adam":
    betas1 = optimizer_param["betas1"]
    betas2 = optimizer_param["betas2"]
    betas = (betas1, betas2)
    eps = optimizer_param["eps"]
    weight_decay = optimizer_param['weight_decay']
    amsgrad = optimizer_param["amsgrad"]


if not data_random_split:
    trainset = data_param["trainset"]
    testset = data_param["testset"]

testing_param = param["testing_param"]
model_path = testing_param["model_path"]
test_path = testing_param["test_path"]
if task == "segmentation":
    test_save_path = testing_param["test_save_path"]
if task == "classification":
    labeled = testing_param["labeled"]
