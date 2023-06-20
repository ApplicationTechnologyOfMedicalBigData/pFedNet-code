import os
NOWPATH=os.path.dirname(os.path.realpath(__file__))
os.chdir(NOWPATH)

from collections import OrderedDict
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from config import *
import torch.nn.functional as F
import re
import time
import utils.loss
from utils import logs
from typing import Dict,List,Optional,Tuple
import utils.my_founction as myfn
if task == "classification" and model == "DenseNet3D":
    from network.DenseNet3D import DenseNet3d as Net
if task == "classification" and model == "DenseNet":
    from network.DenseNet import DenseNet as Net
if task == "segmentation" and model == "UNet3D":
    from network.UNet3D import UNet3D as Net
if task == "segmentation" and model == "UNet2D":
    from network.UNet import UNet as Net


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


def train(net, trainloader, epochs,client_name,server_round,
          client_proper):
    """Train the network on the training set."""
    
    client_loss=client_proper["Fedloss"]
    server_weight_path=client_proper["server_model_path"]

    global criterion, optimizer
    
    # 优化器
    if optimiz == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr, betas, eps, weight_decay, amsgrad)

    if optimiz == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum, dampening, weight_decay, nesterov)
        

    criterion=utils.loss.SelectLossFn(net=net,server_weight_path=server_weight_path,client_loss=client_loss).get_loss()

    # 评价参数初始化
    correct, total, loss, accuracy, running_loss_ = 0, 0, 0.0, 0.0, 0.0

    # epochs=int(epochs/server_round)+1
    if "epochs" in client_proper.keys():
        epochs=client_proper["epochs"]
    # 迭代训练
    for _ in range(0, epochs):
        
        # 当前训练轮次loss初始化：
        running_loss = 0.0
        pre_all=0.0
        pre_true=0.0
        labels_all=0.0
        # print_fn("#",_)
        for images, labels in trainloader:
            images, labels = images.nan_to_num(0).to(DEVICE), labels.nan_to_num(0).to(DEVICE)

            outputs = net(images.nan_to_num(0))
            pred = torch.argmax(outputs.data, dim=1)
            loss = criterion(outputs, labels)
            running_loss += loss

            pred_one_hot=F.one_hot(pred.flatten())
            labels_one_hot=F.one_hot(labels.flatten())
            if labels_one_hot.shape[1]!=num_classes or pred_one_hot.shape[1]!=num_classes:
                labels_one_hot=torch.cat((labels_one_hot,torch.zeros((labels_one_hot.shape[0],num_classes-labels_one_hot.shape[1]),device=DEVICE)),dim=1)
                pred_one_hot=torch.cat((pred_one_hot,torch.zeros((pred_one_hot.shape[0],num_classes-pred_one_hot.shape[1]),device=DEVICE)),dim=1)
            one_hot_corr=(pred_one_hot == labels_one_hot)*pred_one_hot
            correct += one_hot_corr.sum().item()

            pre_all=pre_all+torch.sum(pred_one_hot,dim=0)
            labels_all=labels_all+torch.sum(labels_one_hot,dim=0)
            pre_true=pre_true+torch.sum(one_hot_corr,dim=0)
            total += labels.size(0)

            # time.sleep(10)
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
            p=pre_true/pre_all
            r=pre_true/labels_all
            f1_single=2*p*r/(p+r)
            f1_single=f1_single.nan_to_num(0)
            f1=torch.sum(f1_single)/len(f1_single)
            accuracy = correct / total
            log.logger.info(
                client_name+" - server_round: {},".format(server_round)+
                " epoch: {}/{},".format(str(_ + 1), str(epochs))+
                " training loss: {},".format(running_loss_)+
                " training accuracy: {},".format(accuracy))
    return running_loss_, accuracy


def test(net, testloader,client_proper=None):
    """Validate the network on the entire test set."""
    # 损失函数
    client_loss=client_proper["Fedloss"]
    server_weight_path=client_proper["server_model_path"]

    global criterion

    criterion=utils.loss.SelectLossFn(net=net,server_weight_path=server_weight_path,client_loss=client_loss).get_loss()

    # 评价参数初始化
    loss, accuracy = 0.0, 0.0
    acc_total,acc_correct=0,0.0
    bacc_total,bacc_correct = 0,0.0
    bacc_accuracy=0.0
    pre_all=0.0
    labels_all=0.0
    pre_true=0.0

    # 迭代测试
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].nan_to_num(0).to(DEVICE), data[1].nan_to_num(0).to(DEVICE)
            outputs = net(images.nan_to_num(0))
            loss += criterion(outputs, labels).item()
            pred = torch.argmax(outputs.data, dim=1)
            acc_correct += (pred == labels).sum().item()
            acc_total += labels.size(0)

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
        acc_accuracy = acc_correct / acc_total
        bacc_accuracy = torch.sum(bacc_correct / bacc_total)/num_classes

        return loss, acc_accuracy, bacc_accuracy

class CPath():
    def __init__(self,round,name) -> None:
        self.round=round
        self.name=name
    def _save_path(self):
        path="./save_model/client/save/round_"+str(self.round)+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    def _send_path(self):
        path="./save_model/client/send/round_"+str(self.round)+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_model_path(self):
        save_model_path=self._save_path()+"weight_"+self.name+"_"+task+"_"+model+".pth"
        return save_model_path
    def save_gradient_model_path(self):
        save_path=self._save_path()+f"gradients_"+self.name+"_"+task+"_"+model+".pth"
        return save_path
    def save_server_model_path(self):
        save_path=self._save_path+"server_model_"+self.name+"_"+task+"_"+model+".pth"
        return save_path
    
    def send_gradient_model_path(self,send_dict_name=None):
        if send_dict_name:
            send_path=self._send_path()+"gradients_"+self.name+"_"+task+"_"+model+"_"+send_dict_name+".pth"
        else:
            send_path=self._send_path()+"gradients_"+self.name+"_"+task+"_"+model+".pth"
        return send_path
    def send_model_path(self,send_dict_name=None):
        if send_dict_name:
            send_path=self._send_path()+"weight_"+self.name+"_"+task+"_"+model+"_"+send_dict_name+".pth"
        else:
            send_path=self._send_path()+"weight_"+self.name+"_"+task+"_"+model+".pth"
        return send_path
    
    
    

class Client():
    def __init__(self,
                 client_name:str,
                 client_proper:dict,
                 client_data:list,
                 server_round:int,
                 send_name_dict:Dict[str,List[str]],
                 ) -> None:
        #we need the name and the data 
        self.client_name=client_name
        self.client_proper=client_proper
        self.trainloader=client_data[0]
        self.testloader=client_data[1]
        self.num_examples=client_data[2]
        self.server_round=server_round
        self.net= Net().to(DEVICE)
        self.send_name_dict=send_name_dict


    
    def set_net_parameters(self,read_path,round,config:list=[]):
        client_path=CPath(round,self.client_name)
        init_params_dict={}
        if self.client_proper["server_send_type"]=="server_model":#init that is not .zip instead of .pth 
            init_params_dict=torch.load(read_path)
            self.net.load_state_dict(init_params_dict, strict=True)
            if round==1:
                client_path.round=round-1
                torch.save(init_params_dict,client_path.save_model_path())

        elif self.client_proper["server_send_type"]=="part_server_add_now_client_model":
            now_save_path=client_path.save_model_path()
            now_params_dict=torch.load(now_save_path)
            init_params_dict=torch.load(read_path)
            for name in init_params_dict:
                now_params_dict[name]=init_params_dict[name]
            init_params_dict=now_params_dict
            self.net.load_state_dict(now_params_dict, strict=True)

        elif self.client_proper["server_send_type"]=="part_server_add_last_client_model":
            client_path.round=round-1
            save_path=client_path.save_model_path()
            now_params_dict=torch.load(save_path)
            init_params_dict=torch.load(read_path)
            for name in init_params_dict:
                now_params_dict[name]=init_params_dict[name]
            init_params_dict=now_params_dict
            self.net.load_state_dict(now_params_dict, strict=True)

        elif self.client_proper["server_send_type"]=="last_client_model":
            client_path.round=round-1
            save_model_path=client_path.save_model_path()
            init_params_dict=torch.load(save_model_path)
            self.net.load_state_dict(init_params_dict, strict=True)

        elif self.client_proper["server_send_type"]=="now_client_model":
            save_model_path=client_path.save_model_path()
            init_params_dict=torch.load(save_model_path)
            self.net.load_state_dict(init_params_dict, strict=True)

        return init_params_dict
    
    
    def save_parameters(self,init_params_dict,config=[]):
        return_path=None
        now_params_dict=self.net.state_dict()
        #save weight
        cpath=CPath(self.server_round,self.client_name)

        if self.client_proper["client_return_type"]=="client_model":
            save_path=cpath.save_model_path()
            torch.save(now_params_dict,save_path)

            if "fit" in config:
                return_path=cpath.send_model_path()
                torch.save(now_params_dict,return_path)

        elif self.client_proper["client_return_type"]=="client_sub_paras_model":
            save_path=cpath.save_model_path()
            server_paras_dict=torch.load(self.client_proper["server_model_path"])
            for name in server_paras_dict:
                now_params_dict[name]=now_params_dict[name]-server_paras_dict[name]
            torch.save(now_params_dict,save_path)

            if "fit" in config:
                return_path=cpath.send_model_path()
                torch.save(now_params_dict,return_path)

        elif self.client_proper["client_return_type"]=="part_client_model":
            save_path=cpath.save_model_path()
            return_paras_dict={}
            for name in self.send_name_dict[self.client_proper["client_return_name"]]:
                return_paras_dict[name]=now_params_dict[name]
            torch.save(now_params_dict,save_path)

            if "fit" in config:
                return_path=cpath.send_model_path(send_dict_name=self.client_proper["client_return_name"])
                torch.save(return_paras_dict,return_path)


        elif self.client_proper["client_return_type"]=="gradients_model":
            gradients_dict={}
            for name in now_params_dict:
                gradients_dict[name]=init_params_dict[name]-now_params_dict[name]
            save_path=cpath.save_model_path()
            torch.save(now_params_dict,save_path)

            if "fit" in config:
                return_path=cpath.send_gradient_model_path()
                torch.save(gradients_dict,return_path)

        elif self.client_proper["client_return_type"]=="part_gradients_model":
            gradients_dict={}
            for name in self.send_name_dict[self.client_proper["client_return_name"]]:
                gradients_dict[name]=init_params_dict[name]-now_params_dict[name]
            save_path=cpath.save_model_path()
            torch.save(now_params_dict,save_path)

            if "fit" in config:
                return_path=cpath.send_gradient_model_path(send_dict_name=self.client_proper["client_return_name"])
                torch.save(gradients_dict,return_path)

        return return_path
    
    def close_net_grads(self):
        if self.client_proper["state_grad"]:
            for name,par in self.net.named_parameters():
                if name  in self.send_name_dict[self.client_proper["server_send_name"]] :
                    par.requires_grad=False
    def reset_net_grad(self):
        if self.client_proper["state_grad"]:
            for par in self.net.parameters():
                par.requires_grad=True

    def fit(self,read_path,config=[]):
        result={"name":self.client_name,"samples num":len(self.trainloader.dataset)}
        init_params_dict=self.set_net_parameters(read_path=read_path,round=self.server_round) #read backward fit round parameters

        self.net.train()
        self.close_net_grads()
        loss, accuracy = train(self.net,self.trainloader,epochs,client_name=self.client_name,
                               server_round=self.server_round,client_proper=self.client_proper)
        self.reset_net_grad()
        return_path=self.save_parameters(init_params_dict,config=["fit"]) #save now round net parameters
        #get trained weights save path
        result["return"] = return_path
        result["acc"]=accuracy
        result["loss"]=loss
        return result

    def evaluate(self,read_path,config:list = []):
        result={"name":self.client_name,"samples num":len(self.testloader.dataset)}
        init_params_dict=self.set_net_parameters(read_path=read_path,round=self.server_round)
        self.save_parameters(init_params_dict,config=["eval"])
        self.net.eval()
        loss, acc1, acc2 = test(self.net, self.testloader,client_proper=self.client_proper)


        if task == "segmentation":
            log.logger.info(
                self.client_name+" - server_round: {},".format(self.server_round)+
                "\033[1;32m test loss: {} \033[0m,".format(loss)+
                "\033[1;32m test iou: {} \033[0m".format(acc1)+
                "\033[1;32m test label_iou: {} \033[0m".format(acc2))
        elif task == "classification":
            log.logger.info(
                self.client_name+" - server_round: {},".format(self.server_round)+
                "\033[1;32m test loss: {} \033[0m,".format(loss)+
                "\033[1;32m test accuracy: {} \033[0m".format(acc1)+
                "\033[1;32m bacc: {} \033[0m".format(acc2))
        result["test_loss"] = loss
        result["test_acc1"] = acc1
        result["test_acc2"] = acc2
        return result







