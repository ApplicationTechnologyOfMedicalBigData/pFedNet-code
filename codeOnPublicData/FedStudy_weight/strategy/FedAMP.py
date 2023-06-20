import os
import torch
from config import task,model,DEVICE
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata

class FedAMPStrategy():
    def __init__(self,net) -> None:
        self.net=net
    def tell_client_calculate_gradients(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m fit avg parameters \033[0m"+"-"*10)
        #init weight
        sp=myfn.SPath(server_round-1,None)
        for client_name in clients_proxy:
            if server_round==1:
                sp.name=client_name
                save_model_path=sp.save_model_path()
                send_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["path"]=send_model_path
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
                clients_proxy[client_name]["client_data"].send2client["Fedloss"]="FedAMPLoss"
                clients_proxy[client_name]["client_data"].send2client["server_model_path"]=send_model_path
                torch.save(self.net.state_dict(),save_model_path)
                torch.save(self.net.state_dict(),send_model_path)
            else:
                sp.name=client_name
                send_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="last_client_model"
                clients_proxy[client_name]["client_data"].send2client["server_model_path"]=send_model_path
                clients_proxy[client_name]["client_data"].send2client["Fedloss"]="FedAMPLoss"
            clients_proxy[client_name]["client"].server_round=server_round
        return clients_proxy
    
    def calculate_u(self,fit_results,server_round):
        weight_list=[]
        name_list=[]
        alpha=0.1
        for result in fit_results:
            file_path=result["return"]
            weight_dict=torch.load(file_path)
            vector=myfn.VctorAndParas().params2onevector(weight_dict)
            weight_list.append(vector)
            name_list.append(result["name"])
        long=len(weight_list)
        u_matrix=torch.zeros((long,long),device=DEVICE)
        for i in range(long):
            for j in range(i+1,long):
                sig=self.dericate(weight_list[i],weight_list[j])
                u_matrix[i,j]=sig
        u_matrix=u_matrix+u_matrix.T
        u_matrix=u_matrix*alpha
        u_namda=(1-torch.sum(u_matrix,dim=0))*torch.eye(long,device=DEVICE)
        u_matrix=u_matrix+u_namda
        weight_tensor=torch.cat(weight_list,dim=1)
        u_weight=torch.mm(weight_tensor,u_matrix)
        names=myfn.VctorAndParas().all_name_list(self.net)
        sp=myfn.SPath(server_round,None)
        for i in range(long):
            new_weight=myfn.VctorAndParas().vector2params(self.net,u_weight[:,i],names)
            sp.name=name_list[i]
            save_model=sp.save_model_path()
            send_model=sp.send_model_path()
            torch.save(new_weight,save_model)
            torch.save(new_weight,send_model)
    
    def dericate(self,vector1,vector2):
        sigma=10
        x=torch.norm((vector1-vector2).float(),p=2)**2/(sigma*len(vector1))
        x.requires_grad=True
        y=1-torch.exp(-1*x)
        y.backward()
        return x.grad

    
    def tell_client_caculate_eval(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m evaluate parameters \033[0m"+"-"*10)
        for client_name in clients_proxy:
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="now_client_model"
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]=None
        return clients_proxy




