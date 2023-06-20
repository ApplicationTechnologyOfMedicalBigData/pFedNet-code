import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import rearrange
from config import DEVICE,num_classes,loss_func,model
import utils.my_founction as myfn
VP=myfn.VctorAndParas()


class SelectLossFn():
    def __init__(self,net=None,server_weight_path=None,client_loss=None) -> None:
        self.net=net
        self.server_weight_path=server_weight_path
        self.client_loss=client_loss
    def net_loss_fn(self):
        loss_fn_dict={}
        n_classes = num_classes
        # 损失函数
        loss_fn_dict["cross_entropy_3d"] = CrossEntropy3D().to(DEVICE)
        loss_fn_dict["cross_entropy_2d"] = CrossEntropy2D().to(DEVICE)
        loss_fn_dict["dice_loss"] = DiceLoss(n_classes).to(DEVICE)
        loss_fn_dict["cross_entropy"] = torch.nn.CrossEntropyLoss().to(DEVICE)
        loss_fn_dict["mse"] = torch.nn.MSELoss().to(DEVICE)
        return loss_fn_dict[loss_func]
    
    def fedprox_loss(self,net,server_weight_path,criterion):
        server_weight=torch.load(server_weight_path)
        criterion=FedProxLoss(net,server_weight,criterion).to(DEVICE)
        return criterion
    
    def fedamp_loss(self,net,server_weight_path,criterion):
        server_weight=torch.load(server_weight_path)
        criterion=FedAMPLoss(net,server_weight,criterion).to(DEVICE)
        return criterion
    
    def ditto_loss(self,net,server_weight_path,client_loss,criterion):
        if client_loss=="Ditto_LocalLoss":
            server_weight=torch.load(server_weight_path)
        else:
            server_weight=None
        criterion=DittoLoss(net,server_weight,client_loss,criterion).to(DEVICE)
        return criterion
    
    def superfed_loss(self,net,server_weight_path,client_loss,criterion):
        if client_loss=="SuPerFed_FedLoss":
            server_weight=[]
            for path in server_weight_path:
                server_weight.append(torch.load(path))
        elif client_loss=="SuPerFed_LocalLoss":
            server_weight=torch.load(server_weight_path)
        criterion=SuPerFedLoss(net,client_loss,server_weight,criterion).to(DEVICE)
        return criterion
        
    
    def get_loss(self):
        if self.client_loss=="FedProxLoss":
            criterion=self.net_loss_fn()
            return self.fedprox_loss(self.net,self.server_weight_path,criterion=criterion)
        elif self.client_loss=="FedAMPLoss":
            criterion=self.net_loss_fn()
            return self.fedamp_loss(self.net,self.server_weight_path,criterion=criterion)
        elif "FedRoD" in self.client_loss:
            criterion=self.net_loss_fn()
            criterion=FedRoDLoss(client_loss=self.client_loss,criterion=criterion).to(DEVICE)
            return criterion
        elif "Ditto" in self.client_loss:
            criterion=self.net_loss_fn()
            criterion=self.ditto_loss(self.net,self.server_weight_path,self.client_loss,criterion=criterion).to(DEVICE)
            return criterion
        elif "SuPerFed" in self.client_loss:
            criterion=self.net_loss_fn()
            criterion=self.superfed_loss(self.net,self.server_weight_path,self.client_loss,criterion=criterion).to(DEVICE)
            return criterion
        else:
            return self.net_loss_fn()



class DiceLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.5):
        "dice_loss_plus_cetr_weighted"
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        # self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        smooth = 0.01  # 防止分母为0
        input1 = F.softmax(input, dim=1)

        target1 = F.one_hot(target, self.n_classes)

        input1 = rearrange(input1, 'b n h w s -> b n (h w s)')
        target1 = rearrange(target1, 'b h w s n -> b n (h w s)')

        input1 = input1[:, 1:, :]
        target1 = target1[:, 1:, :].float()

        inter = torch.sum(input1 * target1)
        union = torch.sum(input1) + torch.sum(target1) + smooth
        dice = 2.0 * inter / union

        loss = F.cross_entropy(input, target)

        total_loss = (1 - self.alpha) * loss + (1 - dice) * self.alpha

        return total_loss


class CrossEntropy3D(nn.Module):
    def forward(self, input, target):
        n, c, h, w, d = input.size()
        target=target.flatten().long()
        target = F.one_hot(target).float()
        if target.shape[1]!=c:
            target=torch.cat((target,torch.zeros((target.shape[0],c-target.shape[1]),device=DEVICE)),dim=1)
        # target = F.one_hot(target.view(target.numel(),1).float())
        # input1 = F.log_softmax(input, dim=1)
        # input_pre = torch.argmax(input.data, dim=1)
        # input1=torch.sum(input,dim=1)*input_pre/c
        # input1 = input1.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
        input1 = input.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
        loss = F.cross_entropy(input1, target)
        return loss
    
class CrossEntropy2D(nn.Module):
    def forward(self, input, target):
        n, c, h, w = input.size()
        target=target.flatten().long()
        target = F.one_hot(target).float()
        if target.shape[1]!=c:
            target=torch.cat((target,torch.zeros((target.shape[0],c-target.shape[1]),device=DEVICE)),dim=1)
        # target = F.one_hot(target.view(target.numel(),1).float())
        # input1 = F.log_softmax(input, dim=1)
        # input_pre = torch.argmax(input.data, dim=1)
        # input1=torch.sum(input,dim=1)*input_pre/c
        # input1 = input1.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
        input1 = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        loss = F.cross_entropy(input1, target)
        return loss

class FedProxLoss(nn.Module):
    def __init__(self,net,server_weight,criterion,u=2) -> None:
        super(FedProxLoss,self).__init__()
        self.server_weight=server_weight
        self.criterion=criterion
        self.u=u
        self.net=net
    def one_long(self,shape):
        one_long=1
        for i in shape:
            one_long=one_long*i
        return one_long
    
    def forward(self,input,target):
        regular_loss=0.0
        count=0.0
        for name,paras in self.net.named_parameters():
            regular_loss=regular_loss+torch.sum((paras-self.server_weight[name])**2)
        loss1=self.criterion(input, target)
        loss2=self.u*regular_loss
        loss=loss1+loss2
        return loss
        
class FedAMPLoss(nn.Module):
    def __init__(self,net,server_weight,criterion,alpha=0.1,namda=0.01) -> None:
        super(FedAMPLoss,self).__init__()
        self.server_weight=server_weight
        self.criterion=criterion
        self.alpha=alpha
        self.namda=namda
        self.net=net
    def forward(self,input,target):
        regular_loss=0.0
        for name,paras in self.net.named_parameters():
            regular_loss=regular_loss+torch.sum((paras-self.server_weight[name])**2)
        loss1=self.criterion(input, target)
        loss2=self.namda*regular_loss/self.alpha
        loss=loss1+loss2
        return loss

class FedRoDLoss(nn.Module):
    def __init__(self,client_loss,criterion,gama=2.0) -> None:
        super(FedRoDLoss,self).__init__()
        self.client_loss=client_loss
        self.criterion=criterion
        self.gama=gama
    def forward(self,input,target):
        if self.client_loss=="FedRoD_BRLoss":
            if "3D" in model:
                n, c, h, w, deep = input.size()
                input_rate=input.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
                input_rate=F.softmax(input_rate)
            else :
                n, c, h, w = input.size()
                input_rate=input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
                input_rate=F.softmax(input_rate)
            n_y=torch.sum(input_rate,dim=0)
            
            d,_=input_rate.shape
            p_y=n_y/d
            g_y=p_y*(n_y**self.gama)/(1-p_y)

            one_hot_label=F.one_hot(target.flatten().long())
            n_c=torch.sum(one_hot_label,dim=0)
            p_c=F.softmax(torch.log(n_c.float()))
            g_c=torch.sum((n_c**self.gama)*p_c/(1-p_c))
            loss=0
            for j in range(len(g_y)):
                loss=loss-torch.log(g_y[j])+torch.log(g_c)
            return loss
        elif self.client_loss=="FedRoD_PERLoss":
            return self.criterion(input,target)
        
class DittoLoss(nn.Module):
    def __init__(self,net,server_weight,client_loss,criterion,namda=1.0) -> None:
        super(DittoLoss,self).__init__()
        self.client_loss=client_loss
        self.criterion=criterion
        self.namda=namda
        self.server_weight=server_weight
        self.net=net
    def forward(self,input,target):
        if self.client_loss=="Ditto_GlobalLoss":
            return self.criterion(input, target)
        elif self.client_loss=="Ditto_LocalLoss":
            regular_loss=0.0
            for name,paras in self.net.named_parameters():
                regular_loss=regular_loss+torch.sum((paras-self.server_weight[name])**2)
            loss1=self.criterion(input, target)
            loss2=self.namda*regular_loss
            loss=loss1+loss2
            return loss


class SuPerFedLoss(nn.Module):
    def __init__(self,net,client_loss,server_weight,criterion,
                 namda=0.5,miu=0.5,v=0.5) -> None:
        super(SuPerFedLoss,self).__init__()
        self.client_loss=client_loss
        self.criterion=criterion
        self.server_weight=server_weight
        self.net=net
        self.namda=namda
        self.miu=miu
        self.v=v
    def forward(self,input,target):
        if self.client_loss=="SuPerFed_FedLoss":
            global_weight=self.server_weight[0]
            local_weight=self.server_weight[1]
            cos_a=0
            cos_f_m=0
            cos_l_m=0
            l2fg=0
            flag=0
            
            # cos=0
            # count=0
            # for name,para in self.net.named_parameters():
            #     cos_l_m=torch.norm(local_weight[name],p=2)
            #     cos_f_m=torch.norm(para,p=2)
            #     if cos_l_m > 1e-5 and cos_f_m > 1e-5:
            #         count+=1
            #         g_sub_f=global_weight[name]-para
            #         cos_a=torch.sum(para*local_weight[name])
            #         cos=cos+(cos_a/(cos_l_m*cos_f_m))**2
            #         l2fg=torch.norm(g_sub_f,p=2)+l2fg
            # cos=cos/count
            # l2fg=l2fg/count

            for name,para in self.net.named_parameters():
                weight1=self.onevector(para)
                weight2=self.onevector(local_weight[name])
                g_sub_f=global_weight[name]-para
                gf_weight=self.onevector(g_sub_f)
                if flag == 0:
                    f_vector=weight1
                    l_vector=weight2
                    gf_vector=gf_weight
                    flag=1
                else:
                    f_vector=torch.cat((f_vector,weight1),dim=0)
                    l_vector=torch.cat((l_vector,weight2),dim=0)
                    gf_vector=torch.cat((gf_vector,gf_weight),dim=0)
                cos_a=torch.sum(para*local_weight[name])+cos_a

            l2fg=torch.norm(gf_vector,p=2)
            cos_f_m=torch.norm(f_vector,p=2)
            cos_l_m=torch.norm(l_vector,p=2)
            cos=(cos_a/(cos_f_m*cos_l_m))**2
            
            loss=self.criterion(input,target)+self.miu*l2fg**2+self.v*cos
        elif self.client_loss=="SuPerFed_LocalLoss":
            cos_a=0
            cos_f_m=0
            cos_l_m=0
            l2fg=0
            flag=0
            for name,para in self.net.named_parameters():
                weight1=self.onevector(para)
                weight2=self.onevector(self.server_weight[name])
                if flag == 0:
                    f_vector=weight1
                    l_vector=weight2
                    flag=1
                else:
                    f_vector=torch.cat((f_vector,weight1),dim=0)
                    l_vector=torch.cat((l_vector,weight2),dim=0)
                cos_a=torch.sum(para*self.server_weight[name])+cos_a
            cos_f_m=torch.norm(f_vector,p=2)
            cos_l_m=torch.norm(l_vector,p=2)
            loss=self.criterion(input,target)+self.v*(cos_a/(cos_f_m*cos_l_m))**2
        return loss
    def onevector(self,weight):
        shape=weight.shape
        long=1
        for i in shape:
            long=long*i
        weight=weight.reshape(long,1)
        return weight
    




