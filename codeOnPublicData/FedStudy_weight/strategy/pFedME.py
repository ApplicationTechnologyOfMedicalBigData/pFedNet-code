import os
import torch
from config import task,model,DEVICE
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata

class pFedMEStrategy():
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
    
    def calculate_weight(self,fit_results,server_round):
        beita=1
        toltal_sample=0.0
        weight_dict={}
        for result in fit_results:
            if toltal_sample==0:
                weight_dict=torch.load(result["return"])
                for name in weight_dict:
                    weight_dict[name]=weight_dict[name]*result["samples num"]
                toltal_sample=toltal_sample+result["samples num"]
            else:
                now_weight_dict=torch.load(result["return"])
                for name in weight_dict:
                    weight_dict[name]=weight_dict[name]+now_weight_dict[name]*result["samples num"]
                toltal_sample=toltal_sample+result["samples num"]
        for name in weight_dict:
            weight_dict[name]=weight_dict[name]/toltal_sample

        sp=myfn.SPath(server_round-1,result["name"])
        last_weight_dict=torch.load(sp.save_model_path())
        for name in last_weight_dict:
            weight_dict[name]=(1-beita)*last_weight_dict[name]+beita*weight_dict[name]

        sp.round=server_round
        for result in fit_results:
            sp.name=result["name"]
            save_model=sp.save_model_path()
            send_model=sp.send_model_path()
            torch.save(weight_dict,save_model)
            torch.save(weight_dict,send_model)

    
    def tell_client_caculate_eval(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m evaluate parameters \033[0m"+"-"*10)
        for client_name in clients_proxy:
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="now_client_model"
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]=None
        return clients_proxy




