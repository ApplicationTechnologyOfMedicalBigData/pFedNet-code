import os
import torch
import torch.nn.functional as F
now_path=os.path.dirname(os.path.realpath(__file__))
os.chdir(now_path)
import torch.optim as optim
from Fedgradient import FedGradient
from network.UNet import UNet as Net
from config import *
from utils.dataset import load_random_split_data as load_data
import re
import numpy as np
import time
import utils.my_founction as myfn
import matplotlib.pyplot as plt
import zipfile
from sklearn import neighbors
from utils.my_founction import Clientdata

net = Net().to(DEVICE)

SEED=torch.randn(size=(4,5),requires_grad=True)


def print_fn(x,s="#"):
    print(str(s)*200)
    print(x)
    print(str(s)*200)



# model_path="/home/yawei/Documents/FL_in_local/save_model/client/server_round:x/model_client_name:client0_classification_DenseNet.zip"
model_path="/home/yawei/Documents/FL_in_local/save_model/server/save/round_1/sub_avg_grad_model_client0_classification_DenseNet.pth"
model_path2="/home/yawei/Documents/FL_in_local/save_model/client/send/round_2/"
model_path3="/home/yawei/Documents/LUNA16/candidateOfnodules/"

file_path="/home/yawei/Documents/FL_in_local/data/compress_update/stc.txt"
dir_path="/home/yawei/Documents/berifen/2_FL_in_local-20230410-ADMM/data/compress_update/save/"
name=["conv","classifier"]
name=["conv"]
# client_model=torch.load(model_path2)
# server_model=torch.load(model_path)
# is_same=myfn.model_same(name,client_model,server_model)
# print(is_same)
# myfn.MyPlot.compara(dir_path,type="client_size")
# myfn.MyPlot.compara(dir_path,type="server_size")
long=0
for name,para in net.named_parameters():
    long=long+myfn.VctorAndParas().get_long(para.shape)
    print(name,long)
# zip_bytes,raw_bytes=myfn.Compress.zipsize(model_path2)

# print(zip_bytes,raw_bytes)
#########################################################################
# clients_proxy=STF.load_client()

# name_sequence=[]
# client_dir=[]
# for i in os.listdir(model_path):
#     if  re.findall("send.*zip",i):
#         client_dir.append(i)
#         name=re.findall(".*(client\d+).*",i)
#         name_sequence.append(name[0])

# weight_dir=[]
# for name in name_sequence:#ranke in same sequence
#     for i in os.listdir(model_path2):
#         if  re.findall("weight.*pth",i) and name in i:
#             weight_dir.append(i)

# gradient_path_list=[model_path+"/"+client for client in client_dir]
# weight_path_list=[model_path+"/"+client for client in weight_dir]
# req_name=["classifier"]

# clusterclient=myfn.ClusterClient(gradient_path_list,weight_path_list,req_name,
#                                  clients_proxy,name_sequence,namda=10.0,eta=0.3)#eta=0.3 


# z=clusterclient.get_z()
# print(z)_test()



# myfn.ClusterClient().get_gradients(clients_path_list)


# myfn.MyPlot.plot_global_acc(file_path)

# myfn.MyPlot.compara(dir_path,type="acc")
#########################################################################

# LONG=3000
# # myfn.save_matrix(1000)
# sigma_path="/home/yawei/Documents/FL_in_local/save_model/matrix/"+f"feature_matrix_{LONG}.zip"
# path=os.walk(model_path)
# for _,_,i in os.walk(model_path):
#     print(i)
# csv_name=re.findall("(.*/)(.+zip)",model_path)
# print(csv_name[0][0])
# feature_dict=torch.load(sigma_path)
# grad_dict=torch.load(model_path2)
# weight_dict=torch.load(model_path)


#########################################################################
#tensor trans to vector examples
grad_path="/home/yawei/Documents/berifen/2_FL_in_local-20230410-ADMM/save_model/client/save/round_1/weight_client0_segmentation_UNet2D.pth"
weight_path="/home/yawei/Documents/berifen/2_FL_in_local-20230410-ADMM/save_model/client/save/round_100/weight_client0_classification_DenseNet.pth"
model_path3="/home/yawei/Documents/berifen/2_FL_in_local-20230410-ADMM/save_model/client/send/round_1/send_grad_client0_segmentation_UNet2D_avg_.zip"
grad_dict=torch.load(grad_path)
weight_dict=torch.load(weight_path)
for name in grad_dict:
    print(name)
    if "norm" in name:
        print(name)
        print(grad_dict[name])

paras_name=[ name for name,_ in net.named_parameters()]
name_list=["conv","norm"]

grad_vector_list=myfn.VctorAndParas().params2vector(grad_dict,name_list)
# myfn.Compress.vector2zip(model_path3,grad_vector_list,name_list)
grad_vector_list=myfn.Compress.zip2vector(model_path3,"avg")
for i in grad_vector_list:
    print(i.shape)

weight_one_vector=myfn.VctorAndParas().params2vector(weight_dict,name_list)

# file_path="./save_model/client/server_round:1/"
# file_name="model_client_name:client0_classification_DenseNet.zip"
# # myfn.Compress().vector2zip(file_path,grad_one_vector)
# vector=myfn.Compress.zip2vector(file_path+file_name)


# print(vector.shape)

#########################################################################
#smooth for all long

# vector_non_admm=myfn.Compress().get_topK(grad_one_vector)

# cluster_grad=myfn.ClusterGradients([],[])#init
# grad_vector=myfn.all_long_cluster(weight_one_vector,grad_one_vector,epoch=10)
# vector=myfn.VctorAndParas().list2vector(grad_vector)
# grad_direction = myfn.STC().get_topK(vector)
# # myfn.plot_defference(vector_non_admm,grad_direction,1)

# non_admm_grad=myfn.VctorAndParas().vector2params(net,vector_non_admm)
# admm_grad=myfn.VctorAndParas().vector2params(net,grad_direction)

# myfn.save_matrix(1000)
# _,sigma,namda,value=myfn.sigma_metrix(10)
# print(namda)
# print(1/value)
############################################################################################
#save as zip

# with open("./non_admm.csv","w") as f:
#     count=0
#     for i in vector_non_admm.cpu().detach().numpy():
#         f.writelines(str(i[0]))
#         f.writelines("\n")
#         # count+=1
#         # if count>100:
#         #     break
# with open("./admm.csv","w") as f:
#     count=0
#     for i in grad_direction.cpu().detach().numpy():
#         f.writelines(str(i[0]))
#         f.writelines("\n")
# with zipfile.ZipFile("non_admm.zip","w",compression=8) as z:
#     z.write("./non_admm.csv")
# with zipfile.ZipFile("admm.zip","w",compression=8) as z:
#     z.write("./admm.csv")

######################################################################
#cluster gradients examples

# print("-"*10+"\033[1;33m start get cluster gradients \033[0m"+"-"*10)
# longs=len(grad_one_vector)
# num=int(longs/LONG)
# vector_long=0
# t=[]
# for i in range(num+1):
#     weight_section=weight_one_vector[i*LONG:(i+1)*LONG]
#     grad_section=grad_one_vector[i*LONG:(i+1)*LONG]
#     vector_long=len(grad_section)
#     if vector_long==LONG:
#         t1=time.time()
#         grads=myfn.ClusterGradients(weight_section,grad_section)
#         grad_one_vector[i*LONG:(i+1)*LONG],value_y=grads.get_cluster_gradients()
#         t2=time.time()
#         t.append(t2-t1)
#         t_avg=sum(t)/len(t)
#         print("\x1b[2A\r")
#         print("-"*int(i*100/(num+1))+">"+
#               "="*(100-int(i*100/(num+1)))+
#               "| process:{}%---".format(int(i*100/(num+1)))+
#               "wait for {} s---".format(int(((num+1)-i)*t_avg)))
    
######################################################################
#all long cluster

# weight_path="/home/yawei/Documents/FL_in_local/save_model/client/round_2/gradients_client0_classification_DenseNet.pth"
# gradient_path="/home/yawei/Documents/FL_in_local/save_model/client/round_2/weight_client0_classification_DenseNet.pth"


# weight_dict=torch.load(weight_path)
# grad_dict=torch.load(gradient_path)
# namelist=["conv"]
# vp=myfn.VctorAndParas()
# weight_one_vector=vp.params2vector(weight_dict,namelist)[0]
# grad_one_vector=vp.params2vector(grad_dict,namelist)[0]


# long=300
# myfn.save_matrix(long)
# weight_section=weight_one_vector[0:long]
# # weight_section=weight_section.reshape(long,1)

# grad_section=grad_one_vector[0:long]
# # grad_section=grad_section.reshape(long,1)

# # np.random.seed(0)
# # y1=np.random.randn(long)
# # y2=np.random.randn(long)
# # weight_section=torch.tensor(y1,device=DEVICE).reshape(long,1)
# # grad_section=torch.tensor(y2,device=DEVICE).reshape(long,1)

# t1=time.time()

# grads=myfn.ClusterGradients(weight_section,grad_section,epoch_num=10,long=long)
# gradients,value_y=grads.get_cluster_gradients()
# v=[i.cpu().detach().numpy() for i in value_y[len(value_y)-1]]
# y=grads.cvx_test()
# x=list(range(len(y)))
# t2=time.time()
# print(t2-t1)

# plt.plot(x,y-v)
# plt.show()




# grads.plot_result(gradients,value_y)


# p,sigma,namda,value=myfn.sigma_metrix(3)

# x=torch.mm(p,sigma)
# x=torch.mm(x,p.T)
# y=torch.mm(namda.T,namda)
# print(sigma)
# print(1/(value)*torch.eye(3,3,device="cuda:0"))
# y_v=torch.inverse(sigma)

# print(y_v)


# longs=0
# for name in weight_dict:
#     long=1
#     for i in weight_dict[name].shape:
#         long=long*i
#     longs=longs+long
#     print("{}:".format(name)+"{}".format(longs))


#########################################################################
#calculate net paramenters
# longs=0
# for name,para in net.named_parameters():
#     long=1
#     for i in para.shape:
#         long=long*i
#     longs=longs+long
#     print(name+" : long-{} ".format(long)+" ,longs-{}".format(longs))



###############################################
#gradients examples
# params_dict2=torch.load(model_path2)
# for name,_ in net.named_parameters():
#     params_dict2[name].requires_grad=True
# for name,para in net.named_parameters():
#     print(para.requires_grad)

# net.load_state_dict(params_dict2,strict=True)
# net.train()
# optimizer = torch.optim.Adam(net.parameters(), lr, betas, eps, weight_decay, amsgrad)
# optimizer=FedGradient(params=net.parameters(),lr=lr, momentum=momentum, 
#                                             dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,  near_clients_weight=[],admm_layer=0)
# for name in params_dict2:
#     print(name)

# print("#"*100)
# for name in params_dict:
#     print(name)
# longs=[]
# for name in params_dict:
# # for name,para in net.named_parameters():
#     # if "features.denseblock3.denselayer16.conv2.weight" in name:
#     #     # print(params_dict[name])
#     #     metrix=params_dict[name]
#     metrix=params_dict[name]
#     dimension=metrix.shape
#     long=1
#     for i in dimension:
#         long=long*i
#     if long not in longs:
#         longs.append(long)
# print(len(longs))
# print(longs)
# print(sum(longs))


#########################################################################
#train examples
# 评价参数初始化
# correct, total, loss, accuracy, running_loss_ = 0, 0, 0.0, 0.0, 0.0

# criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
# # 迭代训练
# for _ in range(0,0):
    
#     # 当前训练轮次loss初始化：
#     running_loss = 0.0
#     # print_fn("#",_)
#     for images, labels in trainloader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         outputs = net(images)
#         pred = torch.argmax(outputs.data, dim=1)
#         loss = criterion(outputs, labels)
#         running_loss += loss
#         correct += (pred == labels).sum().item()
#         total += labels.size(0)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()













