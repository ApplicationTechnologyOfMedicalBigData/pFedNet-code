import os
import torch
from dataclasses import dataclass
from config import DEVICE,task,model
import matplotlib.pyplot as plt
import time
import zipfile
import re
from typing import List,Dict,Optional
import cvxpy as cp
import numpy as np



@dataclass
class Clientdata():
    name:str
    proper:dict
    parameters:list
    send2client:dict


#get sigma matrix,for reflash
def sigma_metrix(long):
    #return feature value "sigma" and feature vector "p" and it's raw matrix "namda"
    namda1=torch.eye(long,long,dtype=float,device=DEVICE)
    namda2=torch.eye(long-1,long-1,dtype=float,device=DEVICE)
    col_zeros=torch.zeros((long-1,1),dtype=float,device=DEVICE)
    row_zeros=torch.zeros((1,long),dtype=float,device=DEVICE)
    namda3=torch.concatenate((col_zeros,namda2),axis=1)
    namda4=torch.concatenate((namda3,row_zeros),axis=0)
    namda=namda1-namda4
    namda_multy=torch.mm(namda.T,namda)
    value,p=torch.linalg.eigh(namda_multy)
    sigma=torch.eye(long,long,dtype=float,device=DEVICE)*value
    namda=namda.double()
    value=value.double()
    p=p.double()
    sigma=sigma.double()
    return p,sigma,namda,value


#write to the path, we can read it, instead of recalculate the matrix
def save_matrix(long):
    dir="/home/yawei/Documents/FL_in_local/save_model/matrix/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    sigma_path=dir+f"feature_matrix_{long}.pth"
    p,sigma,namda,value=sigma_metrix(long)
    sigma_dict={"p":p,"sigma":sigma,"namda":namda,"feature value":value}
    torch.save(sigma_dict,sigma_path)

#build single cov metrix
def cov(trainloader,task = "classification"):
    if task=="classification":
        x=[]
        for data in trainloader:
            x.append(data[0])
        input=torch.cat(x)
        input=torch.tensor(input,device=DEVICE)
        b, c, h, w = input.shape
        x = input- torch.mean(input)
        x = x.view(b * c, h * w)
        cov_matrix = torch.matmul(x.T, x) / x.shape[0]
    elif task=="segmentation":
        x=[]
        for data in trainloader:
            x.append(data[0])
        input=torch.cat(x)
        input=torch.tensor(input,device=DEVICE)
        if "2D" in model:
            b, c, h, w = input.shape
            input1=torch.sum(input,dim=0)/b
            cov_matrix=torch.sum(input1,dim=0)/c
        else:
            b, c, h, w, d = input.shape
            input1=torch.sum(input,dim=0)/b
            input2=torch.sum(input1,dim=0)/c
            cov_matrix=torch.sum(input2,dim=2)/d
        cov_matrix=cov_matrix.nan_to_num(0)
        # torch.cuda.manual_seed(0)
        # mask_matrix=torch.rand((h,w,d),device=DEVICE)>p
        
    return cov_matrix

class KNN():
    def __init__(self,name:List[str],cov_list:List[torch.tensor]) -> None:
        self.name=name
        self.cov_list=cov_list
    #get cov_metrix distance
    def distance_matrix(self)->Dict[str,List[str]]:
        long=len(self.cov_list)
        sub_metrix=torch.zeros((long,sum(range(long))),device=DEVICE)
        end=0
        for i in range(long):
            start=end
            step=long-1-i
            end=start+step
            for j in range(start,end):
                sub_metrix[i][j]=1.0
                num=i+j+1-start
                sub_metrix[num][j]=-1.0
        client_cov=[]
        for cov_data in self.cov_list:
            a,b=cov_data.shape
            client_cov.append(cov_data.reshape(a*b,1))
        cov_tensor=torch.cat(client_cov,dim=1)
        sub_cov=torch.mm(cov_tensor,sub_metrix).abs()
        sum_metrix=torch.ones((1,a*b),device=DEVICE)
        sum_cov=torch.mm(sum_metrix,sub_cov)
        distance_matrix=torch.zeros((long,long),device=DEVICE)
        num=0
        for i in range(long-1):
            for j in range(long-1-i):
                distance_matrix[i][j+1+i]=sum_cov[0][num]
                num+=1
        distance_matrix=distance_matrix.T+distance_matrix
        return distance_matrix

    def get_KNN(self,k:int)->List[List[int]]:
        distance_matrix=self.distance_matrix()
        a,b=distance_matrix.shape
        index_list=[]
        max_num=torch.max(distance_matrix)
        for i in range(a):
            distance_matrix[i][i]=max_num
        for i in range(k):
            index=torch.argmin(distance_matrix,dim=1)
            max_num=torch.max(distance_matrix)
            index_list.append(index)
            for j in range(a):
                distance_matrix[j][index[j]]=max_num
        index_list=torch.cat(index_list).reshape(k,a).T.cpu().detach().numpy()
        return index_list
    
    def match_index(self,name_seq:List[str],name:str)->int:
        index=0
        for index_name in name_seq:
            if index_name == name:
                return index
            index+=1
        raise ValueError("not match value")

    def get_q_matrix(self,near_point_dict:Dict[str,List[str]])->torch.tensor:
        long=len(near_point_dict)
        one_hot_index=torch.zeros((long,long),device=DEVICE)
        row_index=0
        for names in near_point_dict.values():
            col_index=[self.match_index(list(near_point_dict.keys()),name) for name in names]
            one_hot_index[row_index][col_index]=1
            row_index+=1
        k=len(names)
        corner_index=torch.zeros((long,long),device=DEVICE)
        for i in range(long):
            for j in range(long):
                if i<j:
                    corner_index[i][j]=one_hot_index[i][j]
        index_tensor=one_hot_index-corner_index.T
        index_tensor=torch.clamp(index_tensor,min=0)
        sum_num=torch.sum(index_tensor)
        q_matrix=torch.zeros((long,sum_num.int()),device=DEVICE)
        q_index=0
        for i in range(long):
            for j in range(long):
                if index_tensor[i][j]==1:
                    q_matrix[i][q_index]=1
                    q_matrix[j][q_index]=-1
                    q_index+=1
        return q_matrix

class ClusterClient():
    def __init__(self,grad_path_list:List[str],weight_path_list:List[str],req_name:List[str],
                 clinet_proxy:List[Clientdata],name_seq:List[str],
                 namda=1,eta=0.001) -> None:
        self.grad_tensor=VctorAndParas().clientziplist2tensor(grad_path_list,req_name)
        self.weight_tensor=VctorAndParas().clientpthlist2matchtensor(weight_path_list,req_name)
        q_matrix=KNN([],[]).get_q_matrix(self.re_ranke_client(clinet_proxy,name_seq))
        self.q_matrix=q_matrix.T
        self.eye=torch.eye(len(name_seq),device=DEVICE)
        self.g_line,self.g_colume=self.grad_tensor.shape
        self.q_line,self.q_colume=q_matrix.shape
        self.rou=1.0
        self.z=VctorAndParas().clientpthlist2matchtensor(weight_path_list,req_name)
        self.oum=torch.zeros((self.g_line,self.q_colume),device=DEVICE)
        self.w=torch.zeros((self.g_line,self.q_colume),device=DEVICE)
        self.namda=namda
        self.eta=eta
        self.name_seq=name_seq

    def re_ranke_client(self,clinet_proxy:List[Clientdata],name_seq:List[str])->Dict[str,List[str]]:
        near_point={}
        for name in name_seq:
            for client in clinet_proxy:
                if client.name == name:
                    near_point[name]=client.proper["near clients"]
        return near_point
    
    def update_z(self):
        z_aa=torch.mm(self.oum,self.q_matrix)
        z_ab=self.grad_tensor/self.g_colume
        z_ac=torch.mm(self.w,self.q_matrix)
        z_ac=self.rou*z_ac
        z_a=self.eta*(z_aa-z_ab+z_ac)+self.z

        z_bb=torch.mm(self.q_matrix.T,self.q_matrix)
        z_b=self.eye+self.eta*self.rou*z_bb
        z_b=torch.inverse(z_b)

        self.z=torch.mm(z_a,z_b)
    
    def updata_w(self):
        w=[]
        for col in range(self.q_colume):
            q=self.q_matrix[col]
            q=q.view(self.g_colume,1)
            w_ba=torch.mm(self.z,q)
            w_bb=self.oum[:,col]
            w_bb=w_bb.view(self.g_line,1)/self.rou
            w_b=w_ba-w_bb

            w_a=self.namda/(self.rou*w_b).norm(p=2)
            w_a=1-w_a
            w_a=max(w_a,0)

            w_col=w_a*w_b
            w.append(w_col)
        self.w=torch.cat(w,dim=1)
        
    def update_oum(self):
        oum_bb=torch.mm(self.z,self.q_matrix.T)
        self.oum=self.oum+self.rou*(self.w-oum_bb)

    def start_update(self,epoch=50,get_hist=False):
        hist=[]
        for i in range(epoch):
            self.update_z()
            self.updata_w()
            self.update_oum()
            if get_hist==True:
                hist.append(torch.sum((self.weight_tensor-self.z).abs()))
        return self.z,hist
    
    def get_z(self,epoch=50)->Dict[str,torch.tensor]:
        self.start_update(epoch=epoch)
        count=0
        z={}
        for name in self.name_seq:
            z[name]=self.z[:,count]
            count+=1
        return z
    
    def cvxtest(self):
        a,b=self.z.shape
        z = cp.Variable(shape=self.z.shape)
        y_a=self.grad_tensor.cpu().detach().numpy()/b
        i_d=np.ones((1,a))
        i_n=np.ones((b,1))
        q=self.q_matrix.T.cpu().detach().numpy()
        w=self.weight_tensor.cpu().detach().numpy()
        problem = cp.Problem(cp.Minimize( i_d@cp.multiply(y_a,z)@i_n + self.namda*cp.norm1(z @ q) + cp.norm2(z - w)**2)/(2*self.eta))
        problem.solve()
        return z.value

    def plot_test(self,epoch:int=1000)->None:
        t1=time.time()
        cvx_z=self.cvxtest()
        cvx_z=cvx_z.flatten()
        t2=time.time()
        print("cvx time:{}".format(t2-t1))

        z,dis=self.start_update(epoch=epoch,get_hist=True)
        z=z.cpu().detach().numpy().flatten()
        t3=time.time()
        print("torch time:{}".format(t3-t2))
        value=[i.cpu().detach().numpy() for i in dis]
        x=list(range(len(z)))
        v_x=list(range(len(value)))
        plt.plot(v_x,value)
        # p1,=plt.plot(x,z,"b-")
        # p2,=plt.plot(x,cvx_z,"r-")
        # plt.legend([p1,p2],["z","cvx_z"])
        plt.show()
        plt.plot(x,z-cvx_z)
        plt.show()

#redefine print founction for locating the output
def print_fn(x,s="#"):
    print(str(s)*200)
    print(x)
    print(str(s)*200)


#make a gradient section be cluster
class ClusterGradients():
    def __init__(self,weight_paras,gradient_paras,
                 w=0.0,y=0.0,rou=1.0,gama=10.0,sr=0.001,epoch_num=5,long=3000):
        self.w=w
        self.y=y
        self.rou=rou
        self.gama=gama
        self.sr=sr
        self.epoch=epoch_num
        self.long=long
        dir="/home/yawei/Documents/FL_in_local/save_model/matrix/"
        if not os.path.exists(dir):
            os.makedirs(dir)
            save_matrix(3000)
            save_matrix(1000)
        sigma_path=dir+f"feature_matrix_{long}.pth"
        feature_dict=torch.load(sigma_path)
        self.namda=feature_dict["namda"]
        self.p=feature_dict["p"]
        self.sigma=feature_dict["sigma"]
        self.feature_value=feature_dict["feature value"]
        self.weight_paras=weight_paras
        self.gradient_paras=gradient_paras
        self.eye=torch.eye(long,long,device=DEVICE)
        self.y_a=[]
        
    def updata_r(self):
        r_a=torch.mm(self.namda,(self.y-self.weight_paras.double())) -self.w/self.rou-self.rou
        r_b=torch.mm(self.namda,(self.weight_paras.double()-self.y))+self.w/self.rou-self.rou
        r_a=torch.clamp(r_a,min=0)
        r_b=torch.clamp(r_b,min=0)
        self.r=r_a-r_b

    def update_y(self):
        # a,b=self.sigma.shape
        k=self.rou*self.sr*self.gama
        if len(self.y_a)==0.0:
            inverse_value=1/(k*self.feature_value+1)
            y_a=torch.mm(self.p,inverse_value*self.eye)
            self.y_a=torch.mm(y_a,self.p.T)
        y_bb=self.weight_paras.double()-self.sr*self.gradient_paras.double()
        y_baa=k*self.namda.T
        n_mul_w=torch.mm(self.namda,self.weight_paras.double()) 
        y_bab=(self.r+n_mul_w+self.w/self.rou)
        y_ba=torch.mm(y_baa,y_bab)
        y_1=torch.mm(self.y_a,y_ba)
        y_2=torch.mm(self.y_a,y_bb)
        self.y=y_1+y_2

    def update_w(self):
        w_a=torch.mm(self.namda,self.y)
        w_b=torch.mm(self.namda,self.weight_paras.double())
        self.w=self.w+self.rou*(self.r-w_a+w_b)
    
    def get_cluster_gradients(self):
        #optimize in sequence
        value_y=[]
        for _ in range(self.epoch):
            self.updata_r()
            self.update_y()
            self.update_w()
            value_y.append(self.y)
            for _ in range(10):
                torch.cuda.empty_cache()
        # gradients=(self.weight_paras.double()-self.y)/self.sr
        gradients_raw=(self.weight_paras.double()-self.y)/self.sr
        gradients=torch.clamp(gradients_raw,max=torch.max(self.gradient_paras.double()),min=torch.min(self.gradient_paras.double()))
        return gradients,value_y
    
    def cvx_test(self):
        y=cp.Variable(shape=self.gradient_paras.shape)
        namda=self.namda.cpu().detach().numpy()
        weight_paras=self.weight_paras.cpu().detach().numpy()
        gradient_paras=self.gradient_paras.cpu().detach().numpy()
        plt.plot(list(range(len(gradient_paras))),gradient_paras)
        plt.show()
        problem = cp.Problem(cp.Minimize(self.gama*cp.norm1(namda@(y-weight_paras))+
                                         cp.norm2(y-(weight_paras-self.sr*gradient_paras))**2)/(2*self.sr))
        problem.solve()
        return y.value
    
    def plot_result(self,gradients,value_y):
        x=list(range(1,self.epoch-1))
        value=[i[0].cpu().detach().numpy() for i in value_y]
        distance=0
        value=[]
        for i in range(self.epoch-1):
            distance=torch.sum(value_y[i]-value_y[i-1])**2
            value.append(distance.cpu().detach().numpy())
        plt.figure(figsize=(180,180))
        plt.subplot(2,2,1)
        plt.title("convergent")
        plt.plot(x,value[1:])
        gradients=gradients.cpu().detach().numpy()
        grad_section=self.gradient_paras.cpu().detach().numpy()
        x=list(range(len(gradients)))
        plt.subplot(2,2,2)
        plt.title("cluster gradients")
        plt.plot(x,gradients)
        plt.title("cluster gradients")
        plt.subplot(2,2,3)
        plt.title("original gradients")
        plt.plot(x,grad_section)
        plt.subplot(2,2,4)
        plt.title("comparison")
        plt.plot(x,grad_section,"r-")
        plt.plot(x,gradients)
        plt.show()

def limit_value(paras,p):
    # para=torch.nan_to_num(para,0.0)
    a,b=paras.shape
    if a*b:
        grad_sec=torch.quantile(paras.abs().float(),1-p)
        para=torch.clamp(torch.nan_to_num(paras,0.0),min=(-1*grad_sec),max=grad_sec)
    else:
        para=paras
    return para

def all_long_cluster(weight_one_vector:torch.tensor,grad_one_vector:torch.tensor,epoch=5,long=3000,p=0.01)->torch.tensor:
    #calculate all long gradient vector
    print("-"*10+"\033[1;33m start get cluster gradients \033[0m"+"-"*10)
    longs=len(grad_one_vector)
    num=int(longs/long)
    t=[]
    vector_long=1
    start_long=0
    long_step=long
    i=0
    grad_vector=[]
    clustergrad=ClusterGradients([],[],long=long,epoch_num=epoch)
    while vector_long:
        if i%(250/epoch)==0:
            clustergrad=ClusterGradients([],[],long=long_step,epoch_num=epoch)#empty the cach
        # clustergrad.weight_paras=limit_value(weight_one_vector[start_long:start_long+long_step],p)
        # clustergrad.gradient_paras=limit_value(grad_one_vector[start_long:start_long+long_step],p)
        clustergrad.weight_paras=weight_one_vector[start_long:start_long+long_step]
        clustergrad.gradient_paras=grad_one_vector[start_long:start_long+long_step]
        start_long=start_long+long_step
        vector_long=len(clustergrad.gradient_paras)
        if vector_long==long_step:
            t1=time.time()
            gradients,_=clustergrad.get_cluster_gradients()
            grad_vector.append(gradients.cpu().detach().numpy())
            t2=time.time()
            t.append(t2-t1)
            t_avg=sum(t)/len(t)
            i=i+1
            if long_step>1000:
                print("\x1b[2A\r")
                print("-"*int(i*100/(num+1))+">"+
                    "="*(100-int(i*100/(num+1)))+
                    "| process:{}%---".format(int(i*100/(num+1)))+
                    "wait for {} s---".format(int(((num+1)-i)*t_avg)))
        elif long_step>vector_long and vector_long>=1000:
            start_long=start_long-long_step
            long_step=1000
            clustergrad=ClusterGradients([],[],long=long_step,epoch_num=epoch)
        elif vector_long<1000 and vector_long>0:
            save_matrix(vector_long)
            start_long=start_long-long_step
            # clustergrad=ClusterGradients(limit_value(weight_one_vector[start_long:start_long+long_step],p),
            #                             limit_value(grad_one_vector[start_long:start_long+long_step],p),
            #                             long=vector_long,epoch_num=epoch)
            clustergrad=ClusterGradients(weight_one_vector[start_long:start_long+long_step],
                                        grad_one_vector[start_long:start_long+long_step],
                                        long=vector_long,epoch_num=epoch)
            gradients,_=clustergrad.get_cluster_gradients()
            grad_vector.append(gradients.cpu().detach().numpy())
            start_long=start_long+long_step
    print("\x1b[2A\r")
    print("-"*100+">"+"| process:{}%---".format(100))
    return grad_vector




def get_required_names(req:str,paras_dict:Dict[str,torch.tensor])->List[str]:
    names=[]
    for name in paras_dict:
        if req in name:
            names.append(name)
    return names

def model_same(req_name:List[str],model1:Dict[str,torch.tensor],model2:Dict[str,torch.tensor])->bool:
    for i in req_name:
        names=get_required_names(i,model1)
    flag=True
    for model_name in names:
        is_same=model1[model_name]!=model2[model_name]
        is_same=torch.flatten(is_same)
        for j in is_same:
            if j:
                flag=False
    return flag

class VctorAndParas():
    #deal with list,vector and trans into net parameters
    def get_long(self,tensshape):
        long=1
        for i in tensshape:
            long=long*i
        return long

    def params2vectorlist(self,paras_dict:Dict[str,torch.tensor],name_list:List[str],through_match:bool=True)->List[torch.tensor]:
        vector_list=[]
        for i in name_list:
            nowlongs=0
            if through_match:
                names=get_required_names(i,paras_dict)
            else:
                names=i
            for name in names:
                shape=paras_dict[name].shape
                if nowlongs==0:
                    #initialize vector
                    onelong=self.get_long(shape)
                    one_vector=paras_dict[name].reshape([onelong,1])
                    nowlongs=1
                elif shape and nowlongs!=0:
                # elif  nowlongs!=0:
                    onelong=self.get_long(shape)
                    one_tensor=paras_dict[name].reshape([onelong,1])
                    one_vector=torch.cat((one_vector,one_tensor),dim=0)
            vector_list.append(one_vector)
        return vector_list
    
    def params2vector(self,paras_dict:Dict[str,torch.tensor],name_list:List[str])->List[torch.tensor]:
        nowlongs=0
        for name in name_list:
            shape=paras_dict[name].shape
            if nowlongs==0:
                #initialize vector
                onelong=self.get_long(shape)
                one_vector=paras_dict[name].reshape([onelong,1])
                nowlongs=1
            elif shape and nowlongs!=0:
            # elif  nowlongs!=0:
                onelong=self.get_long(shape)
                one_tensor=paras_dict[name].reshape([onelong,1])
                one_vector=torch.cat((one_vector,one_tensor),dim=0)
        return one_vector
    
    def params2onevector(self,paras_dict:Dict[str,torch.tensor]):
        nowlongs=0
        for name in paras_dict:
            shape=paras_dict[name].shape
            if nowlongs==0:
                #initialize vector
                onelong=self.get_long(shape)
                one_vector=paras_dict[name].reshape([onelong,1])
                nowlongs=1
            elif shape and nowlongs!=0:
            # elif  nowlongs!=0:
                onelong=self.get_long(shape)
                one_tensor=paras_dict[name].reshape([onelong,1])
                one_vector=torch.cat((one_vector,one_tensor),dim=0)
        return one_vector
    
    def all_name_list(self,net):
        name_list=[]
        for name in net.state_dict():
            name_list.append(name)
        return name_list

    def vectorlist2params(self,net,vector_list:list,name_list:list,through_match:bool=True)->dict:
        net_state={}
        original_nat_state=net.state_dict()
        for index,i in enumerate(name_list):
            startpos=0
            vector=vector_list[index].float()
            if through_match:
                names=get_required_names(i,original_nat_state)
            else:
                names=i
            for name in names:
                shape=original_nat_state[name].shape
                if shape:
                    long=self.get_long(shape)
                    endpos=startpos+long
                    tensor=vector[startpos:endpos]
                    tensor=tensor.reshape(shape)
                    startpos=endpos
                    net_state[name]=tensor
        return net_state
    
    def vector2params(self,net,vector,name_list:list)->dict:
        net_state={}
        original_nat_state=net.state_dict()
        startpos=0
        for name in name_list:
            shape=original_nat_state[name].shape
            if shape:
                long=self.get_long(shape)
                endpos=startpos+long
                tensor=vector[startpos:endpos]
                tensor=tensor.reshape(shape)
                startpos=endpos
                net_state[name]=tensor
        return net_state
    
    def list2vector(self,mylist):
        vector=torch.tensor(mylist[0])
        for index in range(1,len(mylist)):
            vec_sec=torch.tensor(mylist[index])
            vector=torch.cat((vector,vec_sec),dim=0)
        return vector
    
    def clientziplist2tensor(self,path_list:List[str],req_name:List[str]=[])->List[torch.tensor]:
        vector_list=[Compress.zip2vector(path,req_name=req_name) for path in path_list]
        vectors_tensor=vector_list[0][0]
        for index,gradient in enumerate(vector_list):
            if index:
                vectors_tensor=torch.cat((vectors_tensor,gradient[0]),dim=1)
        return vectors_tensor
    
    def clientpthlist2matchtensor(self,path_list:List[str],req_name:List[str]=[])->List[torch.tensor]:
        para_list=[torch.load(path) for path in path_list]
        for para_index,para_dict in enumerate(para_list):
            vector=self.params2vector(para_dict,req_name)
            if para_index==0:
                vector_tensor=vector[0]
            else:
                vector_tensor=torch.cat((vector_tensor,vector[0]),dim=1)
        return vector_tensor
    
    def clientpthlist2matrix(self,path_list:List[str],req_name:List[str]=[])->torch.tensor:
        para_list=[torch.load(path) for path in path_list]
        for para_index,para_dict in enumerate(para_list):
            vector=self.params2vector(para_dict,req_name)
            if para_index==0:
                vector_tensor=vector
            else:
                vector_tensor=torch.cat((vector_tensor,vector),dim=1)
        return vector_tensor
    

class Compress():
    def get_topK(grad_one_vector:torch.Tensor,p=0.01)->torch.Tensor:
        grad_one_vector=grad_one_vector.to(DEVICE)

        grad_positive=torch.clamp(grad_one_vector,min=0)
        grad_sec=torch.quantile(grad_positive.float(),1-p) 
        # grad_sec=torch.quantile(grad_one_vector.float().abs(),1-p) 

        grad_positive=torch.clamp(grad_one_vector-grad_sec,min=0)
        grad_pos_sum=torch.sum(grad_positive)
        grad_pos_one=grad_positive/(grad_one_vector-grad_sec)
        grad_pos_one=torch.nan_to_num(grad_pos_one,nan=1)

        grad_pos_mean=grad_pos_sum/torch.sum(grad_pos_one)
        grad_pos_mean=torch.nan_to_num(grad_pos_mean,nan=0)
        grad_pos_mean_direction=(grad_pos_mean+grad_sec)*grad_pos_one

        grad_nagitive=torch.clamp(grad_one_vector,max=0)
        grad_sec=torch.quantile(grad_nagitive.float().abs(),1-p)

        grad_nagitive=torch.clamp(grad_one_vector+grad_sec,max=0)
        grad_nag_sum=torch.sum(grad_nagitive) 
        grad_nag_one=grad_nagitive/(grad_one_vector+grad_sec)
        grad_nag_one=torch.nan_to_num(grad_nag_one,nan=1)

        grad_nag_mean=grad_nag_sum/torch.sum(grad_nag_one)
        grad_nag_mean=torch.nan_to_num(grad_nag_mean,nan=0)
        grad_nag_mean_direction=(grad_nag_mean-grad_sec)*grad_nag_one

        grad_direction=grad_pos_mean_direction+grad_nag_mean_direction

        return grad_direction
    
    def vector2zip(file_path:str,vector_list:List[torch.tensor],name_list:List[str])->str:
        if os.path.exists(file_path):
            z=zipfile.ZipFile(file_path,"w",compression=8)
            z.close()
        path=re.findall("(.*/)(.+zip)",file_path)
        file_path=path[0][0]
        zip_name=path[0][1]
        for index,vector in enumerate(vector_list):
            # csv_name=re.findall("(.+).zip",zip_name)[0]+f"{index}"+".csv"
            csv_name=re.findall("(.+).zip",zip_name)[0]+"_"+name_list[index]+".csv"
            with open(file_path+csv_name,"w") as f:
                for i in vector.cpu().detach().numpy():
                    f.writelines(str(i[0]))
                    f.writelines("\n")
            with zipfile.ZipFile(file_path+zip_name,"a",compression=8) as z:
                z.write(file_path+csv_name,csv_name)
            os.system("rm "+file_path+csv_name)
        return file_path+zip_name
    
    def onetensor2zip(file_path:str,onetensor)->str:
        if os.path.exists(file_path):
            z=zipfile.ZipFile(file_path,"w",compression=8)
            z.close()
        path=re.findall("(.*/)(.+zip)",file_path)
        file_path=path[0][0]
        zip_name=path[0][1]
        csv_name=re.findall("(.+).zip",zip_name)[0]+".csv"
        with open(file_path+csv_name,"w") as f:
            for vector in onetensor.cpu().detach().numpy():
                f.writelines(str(vector))
                f.writelines("\n")
        with zipfile.ZipFile(file_path+zip_name,"a",compression=8) as z:
            z.write(file_path+csv_name,csv_name)
        os.system("rm "+file_path+csv_name)
        return file_path+zip_name

    def zip2vector(file_path:str,req_name:List[Optional[str]]=[])->List[torch.tensor]:
        path=re.findall("(.*/)(.+zip)",file_path)
        file_path=path[0][0]
        zip_name=path[0][1]
        with zipfile.ZipFile(file_path+zip_name,"r",compression=8) as z:
            vector_list=[]
            for filename in z.filelist:
                if req_name:
                    flag=1
                    for req in req_name:
                        if req in filename.filename:
                            flag=0
                    if flag:
                        continue
                name=z.extract(filename.filename,file_path)
                with open(name,"r") as f:
                    one_vector_list=[]
                    for i in f.readlines():
                        one_vector_list.append(float(i))
                    os.system("rm "+name)
                vector=torch.tensor(one_vector_list,device=DEVICE).reshape(len(one_vector_list),1)
                vector_list.append(vector)
        return vector_list
    
    def zipsize(file_path):
        files=os.listdir(file_path)
        zip_bytes=0
        raw_bytes=0
        for file in files:
            with zipfile.ZipFile(file_path+file,"r",compression=8) as z:
                for filename in z.namelist():
                    info=z.getinfo(filename)
                    zip_bytes=zip_bytes+info.compress_size
                    raw_bytes=raw_bytes+info.file_size
        return zip_bytes/1024,raw_bytes/1024
        
class SPath():
    def __init__(self,round,client_name) -> None:
        self.round=round
        self.name=client_name
    def _save_path(self):
        path="./save_model/server/save/"+"round_"+str(self.round)+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    def _send_path(self):
        path="./save_model/server/send/"+"round_"+str(self.round)+"/"
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    def save_model_path(self):
        save_model_path=self._save_path()+"model_"+self.name+"_"+task+"_"+model+".pth"
        return save_model_path
    def send_model_path(self):
        send_model_pth=self._send_path()+"model_"+self.name+"_"+task+"_"+model+".pth"
        return send_model_pth
    def save_paras_zip_path(self):
        save_zip_pth=self._save_path()+"params_"+self.name+"_"+task+"_"+model+".zip"
        return save_zip_pth
    def send_avg_grad_path(self):
        send_grad_path=self._send_path()+"avg_grad_"+self.name+"_"+task+"_"+model+".zip"
        return send_grad_path
    def sub_avg_grad_model_path(self):
        now_model_path=self._save_path()+"sub_avg_grad_model_"+self.name+"_"+task+"_"+model+".pth"
        return now_model_path
    def send_personal_grad_path(self):
        send_zip_path=self._send_path()+"personal_grad_"+self.name+"_"+task+"_"+model+".zip"
        return send_zip_path



