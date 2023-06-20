import os
NOWPATH=os.path.dirname(os.path.realpath(__file__))
os.chdir(NOWPATH)

from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from config import *
import re
import time
import utils.loss
from utils import logs
import sys
from typing import Dict,List,Optional,Tuple
import torch.nn.functional as F
import utils.my_founction as myfn
if task == "classification" and model == "DenseNet3D":
    from network.DenseNet3D import DenseNet3d as Net

if task == "classification" and model == "DenseNet":
    from network.DenseNet import DenseNet as Net

if task == "segmentation" and model == "UNet3D":
    from network.UNet3D import UNet3D as Net

if task == "segmentation" and model == "UNet2D":
    from network.UNet import UNet as Net




#get every client number or name
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, default='./config/DenseNet.json', help='config path')
# parser.add_argument('--now_client_num', type=str, default=0)
# args = parser.parse_args()
# now_client_num= args.now_client_num


# 可视化
visual_path = os.path.join("./logs", "visualization")
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
writer = SummaryWriter(visual_path)




log=logs.client_log(NOWPATH,task,model)

def display_para_name(client_name,net):
    # read name more detail
    if client_name == "client0":
        count=0
        for name,para in net.named_parameters():
            myfn.print_fn(name)
            myfn.print_fn(re.findall("weight",name))
            myfn.print_fn("weight" in name)
            count+=1
            # if "denseblock" in name:
            print("{} : name : ".format(count)+name)



def train(net, trainloader, epochs,client_name,server_round):
    """Train the network on the training set."""
    # 损失函数
    
    global criterion, optimizer
    if loss_func == "cross_entropy_3d":
        if "3D" in model:
            criterion = utils.loss.CrossEntropy3D().to(DEVICE)
        elif "2D" in model:
            criterion = utils.loss.CrossEntropy2D().to(DEVICE)

    if loss_func == "dice_loss":
        n_classes = num_classes
        criterion = utils.loss.DiceLoss(n_classes).to(DEVICE)

    if loss_func == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    if loss_func == "mse":
        criterion = torch.nn.MSELoss().to(DEVICE)

    # 优化器
    if optimiz == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr, betas, eps, weight_decay, amsgrad)

    if optimiz == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum, dampening, weight_decay, nesterov)

    if optimiz == "FedSgd":
        import Fedgradient 
        # display_para_name(client_name,net)
        optimizer = Fedgradient.FedGradient(params=net.parameters(),lr=lr, momentum=momentum, 
                                            dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,  near_clients_weight=[],admm_layer=0)



    
    # 评价参数初始化
    correct, total, loss, accuracy, running_loss_ = 0, 0, 0.0, 0.0, 0.0

    # epochs=int(epochs/server_round)+1
    # 迭代训练
    for _ in range(0, epochs):
        
        # 当前训练轮次loss初始化：
        running_loss = 0.0
        pre_all=0.0
        pre_true=0.0
        labels_all=0.0
        # print_fn("#",_)
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = net(images.nan_to_num(0))
            pred = torch.argmax(outputs.data, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pred_one_hot=F.one_hot(pred.flatten())
            labels_one_hot=F.one_hot(labels.flatten())
            if labels_one_hot.shape[1]!=num_classes or pred_one_hot.shape[1]!=num_classes:
                labels_one_hot=torch.cat((labels_one_hot,torch.zeros((labels_one_hot.shape[0],num_classes-labels_one_hot.shape[1]),device=DEVICE)),dim=1)
                pred_one_hot=torch.cat((pred_one_hot,torch.zeros((pred_one_hot.shape[0],num_classes-pred_one_hot.shape[1]),device=DEVICE)),dim=1)
            one_hot_corr=(pred_one_hot == labels_one_hot)*pred_one_hot
            # correct += one_hot_corr.sum().item()

            pre_all=pre_all+torch.sum(pred_one_hot,dim=0)
            labels_all=labels_all+torch.sum(labels_one_hot,dim=0)
            pre_true=pre_true+torch.sum(one_hot_corr,dim=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        running_loss_ = running_loss / len(trainloader)

        if task == "segmentation":
            dice=2*torch.sum(pre_true)/torch.sum(pre_all+labels_all)
            iou=dice/(2-dice)

            max_index=torch.argmax(labels_all)
            index_matrix=torch.ones((1,num_classes),device=DEVICE)
            index_matrix[:,max_index]=0
            label_dice=2*torch.sum(pre_true*index_matrix)/torch.sum((pre_all+labels_all)*index_matrix)
            label_iou=label_dice/(2-label_dice)
            accuracy=label_iou
            log.logger.info(
                client_name+" - server_round: {},".format(server_round)+
                " epoch: {}/{},".format(str(_ + 1), str(epochs))+
                " training loss: {},".format(running_loss_)+
                " training iou: {},".format(iou)+
                " training label_iou: {},".format(label_iou))

        if task == "classification":
            accuracy = correct / total
            log.logger.info(
                client_name+" - server_round: {},".format(server_round)+
                " epoch: {}/{},".format(str(_ + 1), str(epochs))+
                " training loss: {},".format(running_loss_)+
                " training accuracy: {},".format(accuracy))
    return running_loss_, accuracy


def test(net, testloader):
    """Validate the network on the entire test set."""
    # 损失函数
    global criterion
    if loss_func == "cross_entropy_3d":
        if "3D" in model:
            criterion = utils.loss.CrossEntropy3D().to(DEVICE)
        elif "2D" in model:
            criterion = utils.loss.CrossEntropy2D().to(DEVICE)

    if loss_func == "dice_loss":
        n_classes = num_classes
        criterion = utils.loss.DiceLoss(n_classes).to(DEVICE)

    if loss_func == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    if loss_func == "mse":
        criterion = torch.nn.MSELoss().to(DEVICE)
    # 评价参数初始化
    correct, total, loss, accuracy = 0, 0, 0.0, 0.0
    # 迭代测试
    loss, accuracy = 0.0, 0.0
    bacc_total,bacc_correct = 0,0.0
    pre_all=0.0
    labels_all=0.0
    pre_true=0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs.data, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if task == "classification":
                pred_one_hot=F.one_hot(pred)
                label_correct=(pred == labels).view(len(labels),1)
                bacc_correct = label_correct*pred_one_hot
                bacc_correct=torch.sum(bacc_correct,dim=0)
                bacc_total = torch.sum(pred_one_hot,dim=0)+1e-6
            elif task == "segmentation":
                pred_one_hot=F.one_hot(pred.flatten())
                labels_one_hot=F.one_hot(labels.flatten())
                if labels_one_hot.shape[1]!=num_classes or pred_one_hot.shape[1]!=num_classes:
                    labels_one_hot=torch.cat((labels_one_hot,torch.zeros((labels_one_hot.shape[0],num_classes-labels_one_hot.shape[1]),device=DEVICE)),dim=1)
                    pred_one_hot=torch.cat((pred_one_hot,torch.zeros((pred_one_hot.shape[0],num_classes-pred_one_hot.shape[1]),device=DEVICE)),dim=1)
            
                one_hot_corr=(pred_one_hot == labels_one_hot)*pred_one_hot

                pre_all=pre_all+torch.sum(pred_one_hot,dim=0)
                labels_all=labels_all+torch.sum(labels_one_hot,dim=0)
                pre_true=pre_true+torch.sum(one_hot_corr,dim=0)

    loss = loss / len(testloader)

    if task == "segmentation":
        dice=2*torch.sum(pre_true)/torch.sum(pre_all+labels_all)
        iou=dice/(2-dice)

        max_index=torch.argmax(labels_all)
        index_matrix=torch.ones((1,num_classes),device=DEVICE)
        index_matrix[:,max_index]=0
        label_dice=2*torch.sum(pre_true*index_matrix)/torch.sum((pre_all+labels_all)*index_matrix)
        label_iou=label_dice/(2-label_dice)
        return loss, iou, label_iou

    if task == "classification":
        accuracy = correct / total

        return loss, accuracy,bacc_total

class Client():
    def __init__(self,
                 client_name:str,
                 client_proper:dict,
                 client_data:list,
                 server_round:int,
                 send_name_dict:Dict[str,List[str]],
                #  net:Net,
                 ) -> None:
        #we need the name and the data 
        self.client_name=client_name
        self.client_proper=client_proper
        # self.near_clients_name=client_proper["near clients"]
        self.trainloader=client_data[0]
        self.testloader=client_data[1]
        self.num_examples=client_data[2]
        self.server_round=server_round
        self.net= Net().to(DEVICE)
        self.send_name_dict=send_name_dict
    
    def set_net_parameters(self,read_path,round,config:list=[]):
        if self.client_proper["state"]=="personal" or self.client_proper["state"]=="eval":
            save_path = "./save_model/client/save/round_"+str(round)+"/"
            save_model_path=save_path+"weight_"+self.client_name+"_"+task+"_"+model+".pth"

            vector_list=myfn.Compress.zip2vector(read_path)

            if self.client_proper["state"]=="personal":
                update_now_dict_name="avg"
            elif self.client_proper["state"]=="eval":
                update_now_dict_name="personal"
            for name in self.send_name_dict:
                if name==update_now_dict_name:
                    name_list=self.send_name_dict[name]
            grad_dict=myfn.VctorAndParas().vector2params(self.net,vector_list,name_list)
            #set now net parameters
            init_params_dict=torch.load(save_model_path)

            #required parameters should be sub,personalized parameters should remain init
            for grad_name in grad_dict:
                init_params_dict[grad_name]=init_params_dict[grad_name]-grad_dict[grad_name]
            self.net.load_state_dict(init_params_dict, strict=True)

        elif self.client_proper["state"]=="init":#init that is not .zip instead of .pth 
            init_params_dict=torch.load(read_path)
            self.net.load_state_dict(init_params_dict, strict=True)

        elif self.client_proper["state"]=="avg":
            save_path = "./save_model/client/save/round_"+str(round-1)+"/"
            save_model_path=save_path+f"weight_{self.client_name}_{task}_{model}.pth"
            init_params_dict=torch.load(save_model_path)
            self.net.load_state_dict(init_params_dict, strict=True)
        return init_params_dict
    
    def save_compress(self,init_params_dict:dict,gradients_dict:dict,name_list:list,path:str,type:str="stc")->str:
        #compress data
        grad_direction_list=[]
        grad_vector_list=myfn.VctorAndParas().params2vector(gradients_dict,name_list)
        weight_one_vector=myfn.VctorAndParas().params2vector(init_params_dict,name_list)
        if type=="stc":
            for index in range(len(grad_vector_list)):
                grad_direction_list.append(myfn.Compress.get_topK(grad_vector_list[index],p=0.2))
        elif type=="cluster_stc":
            for index in range(len(grad_vector_list)):
                grad_vector=myfn.all_long_cluster(weight_one_vector[index],grad_vector_list[index],epoch=20,gama=self.client_proper["gama"])
                vector=myfn.VctorAndParas().list2vector(grad_vector)
                grad_direction_list.append(myfn.Compress.get_topK(vector,p=0.2))
        else:
            for index in range(len(grad_vector_list)):
                grad_direction_list.append(grad_vector_list[index])
        myfn.Compress.vector2zip(path,grad_direction_list,name_list)
    
    def send_parameters(self,init_params_dict,config=["compress"],compress_type="stc"):
        save_path = "./save_model/client/send/round_"+str(self.server_round)+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        send_path=None
        #caculate gradients after several epochs
        gradients_dict={}
        now_params_dict=self.net.state_dict()
        for name in now_params_dict:
            gradients_dict[name]=init_params_dict[name]-now_params_dict[name]
        if "compress" in config:
            #send compress data path
            if self.client_proper["state"]=="init":
                send_dict_name="avg"
            else:
                send_dict_name=self.client_proper["state"]
            send_path=save_path+f"send_grad_{self.client_name}_{task}_{model}_"+send_dict_name+"_.zip"
            self.save_compress(init_params_dict,gradients_dict,self.send_name_dict[send_dict_name],send_path,type=compress_type)
        return send_path

    def save_parameters(self,init_params_dict,config=[]):
        save_path = "./save_model/client/save/round_"+str(self.server_round)+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_model_path,save_gradients_path=None,None
        now_params_dict=self.net.state_dict()
        #save weight
        save_model_path=save_path+f"weight_{self.client_name}_{task}_{model}.pth"
        #save init parameters which are mantained to sub next turn
        fit_name=[]
        for name in self.send_name_dict:
            fit_name=fit_name+self.send_name_dict[name]
        for i in fit_name:
            names=myfn.get_required_names(i,now_params_dict)
            for save_name in names:
                now_params_dict[save_name]=init_params_dict[save_name]
        torch.save(now_params_dict,save_model_path)

        if "gradient" in config:
            gradients_dict={}
            for name in now_params_dict:
                gradients_dict[name]=init_params_dict[name]-now_params_dict[name]
            #back up model parameters and gradient data
            save_gradients_path=save_path+f"gradients_{self.client_name}_{task}_{model}.pth"
            torch.save(gradients_dict,save_gradients_path)

        return save_model_path,save_gradients_path

    def fit(self,read_path,config=["all"]):
        result={"name":self.client_name,"samples num":len(self.trainloader.dataset)}
        
        init_params_dict=self.set_net_parameters(read_path=read_path,round=self.server_round,config=[]) #read backward fit round parameters
        self.net.train()
        loss, accuracy = train(self.net,self.trainloader,epochs,client_name=self.client_name,
                               server_round=self.server_round)
        
        compress_type=None
        #compress type
        if "stc" in config:
            compress_type="stc"
        elif "cluster_stc" in config:
            compress_type="cluster_stc"
        
        save_model_path,save_gradients_path=self.save_parameters(init_params_dict) #save now round net parameters
        zip_path=self.send_parameters(init_params_dict,compress_type=compress_type) #save now round net parameters
        
        #get trained weights save path
        flag=0
        if "all" in config:
            flag=1
        if "weights" in config or flag==1:
            result["weights"] = save_model_path
        if "gradients" in config or flag==1:
            result["gradients"] = save_gradients_path
        if "compress" in config or flag == 1:
            result["compress"] = zip_path
        if "loss" in config or flag==1:
            result["last train loss"] = loss
        if "accuracy" in config or flag==1:
            result["last train accuracy"] = accuracy
        return result

    def evaluate(self,read_path,config:list = []):
        result={"name":self.client_name,"samples num":len(self.testloader.dataset)}
        init_params_dict=self.set_net_parameters(read_path=read_path,round=self.server_round,config=[])
        self.save_parameters(init_params_dict)
        self.net.eval()
        loss, acc1,acc2 = test(self.net, self.testloader)
        log.logger.info(
            self.client_name+" - server_round: {},".format(self.server_round)+
            "\033[1;32m test loss: {} \033[0m,".format(loss)+
            "\033[1;32m test acc1: {} \033[0m".format(acc1)+
            "\033[1;32m test acc2: {} \033[0m".format(acc2))
        if "all" in config:
            result["test loss"] = loss
            result["test acc1"] = acc1
            result["test acc2"] = acc2
        else:
            if "loss" in config:
                result["test loss"] = loss
            elif "accuracy" in config:
                result["test accuracy"] = accuracy
        return result







