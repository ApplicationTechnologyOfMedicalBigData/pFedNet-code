import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn

class SuPerFedStrategy():
    def __init__(self,net) -> None:
        self.net=net
    def tell_client_calculate_FedWeight(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m fit FedWeight \033[0m"+"-"*10)
        #init weight
        sp=myfn.SPath(server_round-1,None)
        for client_name in clients_proxy:
            if server_round==1:
                sp.name=client_name+"_GlobalWeight"
                save_global_model_path=sp.save_model_path()
                send_global_model_path=sp.send_model_path()
                sp.name=client_name+"_LocalWeight"
                save_local_model_path=sp.save_model_path()
                send_local_model_path=sp.send_model_path()
                sp.name=client_name+"_FedWeight"
                save_fed_model_path=sp.save_model_path()
                send_fed_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["path"]=send_global_model_path
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
                clients_proxy[client_name]["client_data"].send2client["Fedloss"]="SuPerFed_FedLoss"
                clients_proxy[client_name]["client_data"].send2client["server_model_path"]=[send_global_model_path,send_local_model_path]
                torch.save(self.net.state_dict(),save_global_model_path)
                torch.save(self.net.state_dict(),send_global_model_path)
                torch.save(self.net.state_dict(),save_local_model_path)
                torch.save(self.net.state_dict(),send_local_model_path)
                torch.save(self.net.state_dict(),save_fed_model_path)
                torch.save(self.net.state_dict(),send_fed_model_path)
            else:
                sp.name=client_name+"_LocalWeight"
                send_Fedmodel_path=sp.send_model_path()
                sp.name=client_name+"_GlobalWeight"
                send_Globalmodel_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="last_client_model"
                clients_proxy[client_name]["client_data"].send2client["server_model_path"]=[send_Globalmodel_path,send_Fedmodel_path]
                clients_proxy[client_name]["client_data"].send2client["Fedloss"]="SuPerFed_FedLoss"
            clients_proxy[client_name]["client"].server_round=server_round
        return clients_proxy
    
    def calculate_Globalweight(self,fit_results,server_round):

        for result in fit_results:
            client_name=result["name"]+"_FedWeight"
            sp=myfn.SPath(round=server_round,client_name=client_name)
            save_path=sp.save_model_path()
            torch.save(torch.load(result["return"]),save_path)
            send_path=sp.send_model_path()
            torch.save(torch.load(result["return"]),send_path)

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

        sp=myfn.SPath(server_round,None)
        for result in fit_results:
            sp.name=result["name"]+"_GlobalWeight"
            save_model=sp.save_model_path()
            send_model=sp.send_model_path()
            torch.save(weight_dict,save_model)
            torch.save(weight_dict,send_model)


    def tell_client_calculate_LocalWeight(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m fit LocalWeight \033[0m"+"-"*10)
        sp=myfn.SPath(server_round-1,None)
        for client_name in clients_proxy:
            sp.name=client_name+"_LocalWeight"
            send_local_model_path=sp.send_model_path()
            sp.name=client_name+"_FedWeight"
            send_Fed_model_path=sp.send_model_path()
            clients_proxy[client_name]["client_data"].send2client["path"]=send_local_model_path
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
            clients_proxy[client_name]["client_data"].send2client["Fedloss"]="SuPerFed_LocalLoss"
            clients_proxy[client_name]["client_data"].send2client["server_model_path"]=send_Fed_model_path
        return clients_proxy

    def calculate_Loaclweight(self,fit_results,server_round,namda=0.5):
        sp=myfn.SPath(round=server_round-1,client_name=None)
        for result in fit_results:
            
            sp.name=result["name"]+"_FedWeight"
            fed_weight=torch.load(sp.save_model_path()) 

            sp.round=server_round
            sp.name=result["name"]+"_LocalWeight"
            save_local_weight_path=sp.save_model_path()
            local_weight=torch.load(result["return"])
            torch.save(local_weight,save_local_weight_path)
            send_local_weight_path=sp.send_model_path()
            torch.save(local_weight,send_local_weight_path)

            eval_weight={}
            for name in fed_weight:
                eval_weight[name]=(1-namda)*fed_weight[name]+namda*local_weight[name]
            
            sp.name=result["name"]+"_EvalWeight"
            save_eval_weight_path=sp.save_model_path()
            torch.save(eval_weight,save_eval_weight_path)
            send_eval_weight_path=sp.send_model_path()
            torch.save(eval_weight,send_eval_weight_path)


    def tell_client_caculate_eval(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m evaluate parameters \033[0m"+"-"*10)
        sp=myfn.SPath(server_round,None)
        for client_name in clients_proxy:
            sp.name=client_name+"_EvalWeight"
            send_model_path=sp.send_model_path()
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
            clients_proxy[client_name]["client_data"].send2client["path"]=send_model_path
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]=None
        return clients_proxy




