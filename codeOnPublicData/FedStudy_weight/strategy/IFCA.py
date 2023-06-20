import os
import torch
from config import epochs
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata

class IFCAStrategy():
    def __init__(self,net) -> None:
        self.net=net
    def tell_client_calculate_weight(self,server_round,clients_proxy,cluster_num):
        print("-"*10+"\033[1;33m fit avg parameters \033[0m"+"-"*10)
        #init weight
        client_name_list=list(clients_proxy.keys())
        sp=myfn.SPath(server_round-1,None)
        for client_name in clients_proxy:
            if server_round==1:
                sp.name=client_name
                save_model_path=sp.save_model_path()
                send_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["path"]=send_model_path
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
                clients_proxy[client_name]["client_data"].proper["cluster_index"]=self._get_init_cluster(client_name_list,client_name,cluster_num)
                torch.save(self.net.state_dict(),save_model_path)
                torch.save(self.net.state_dict(),send_model_path)
            else:
                sp.name=client_name
                send_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
                clients_proxy[client_name]["client_data"].send2client["epochs"]=epochs
                clients_proxy[client_name]["client_data"].send2client["path"]=send_model_path
            clients_proxy[client_name]["client"].server_round=server_round
        return clients_proxy
    
    def calculate_weight(self,fit_results,server_round,conmunicate_name,clients_proxy,cluster_index):
        toltal_sample=0.0
        cluster_total_sample=0.0
        weight_dict={}
        for result in fit_results:
            if toltal_sample==0:
                avg_weight_dict=torch.load(result["return"])
                for name in conmunicate_name["avg"]:
                    avg_weight_dict[name]=avg_weight_dict[name]*result["samples num"]
                toltal_sample=toltal_sample+result["samples num"]
            else:
                now_avg_weight_dict=torch.load(result["return"])
                for name in conmunicate_name["avg"]:
                    avg_weight_dict[name]=avg_weight_dict[name]+now_avg_weight_dict[name]*result["samples num"]
                toltal_sample=toltal_sample+result["samples num"]

            if clients_proxy[result["name"]]["client_data"].proper["cluster_index"]==cluster_index:
                if cluster_total_sample==0:
                    per_weight_dict=torch.load(result["return"])
                    for name in conmunicate_name["personal"]:
                        per_weight_dict[name]=per_weight_dict[name]*result["samples num"]
                    cluster_total_sample=cluster_total_sample+result["samples num"]
                else:
                    now_per_weight_dict=torch.load(result["return"])
                    for name in conmunicate_name["personal"]:
                        per_weight_dict[name]=per_weight_dict[name]+now_per_weight_dict[name]*result["samples num"]
                    cluster_total_sample=cluster_total_sample+result["samples num"]
            

        for name in conmunicate_name["avg"]:
            weight_dict[name]=avg_weight_dict[name]/toltal_sample
        if cluster_total_sample!=0:
            for name in conmunicate_name["personal"]:
                weight_dict[name]=per_weight_dict[name]/cluster_total_sample
        
        sp=myfn.SPath(server_round-1,None)
        sp.name="cluster"+str(cluster_index)
        if cluster_total_sample==0:
            per_weight_dict=torch.load(sp.save_model_path())
            for name in conmunicate_name["personal"]:
                weight_dict[name]=per_weight_dict[name]
        sp.round=server_round
        save_model=sp.save_model_path()
        torch.save(weight_dict,save_model)
        send_model=sp.send_model_path()
        torch.save(weight_dict,send_model)

    
    def tell_client_estimate_loss(self,server_round,clients_proxy,cluster_index):
        print("-"*10+"\033[1;33m estimate cluster{} loss \033[0m".format(cluster_index)+"-"*10)
        for client_name in clients_proxy:
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
            clients_proxy[client_name]["client_data"].send2client["path"]=myfn.SPath(server_round,"cluster"+str(cluster_index)).send_model_path()
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
            clients_proxy[client_name]["client_data"].send2client["epochs"]=1
        return clients_proxy
    
    def flush_cluster_index(self,acc_list,loss_list,clients_proxy,server_round,test_results):
        name_list=[result["name"] for result in test_results]
        acc=[]
        loss=[]
        for i in loss_list:
            d=len(i)
            loss.append(torch.tensor(i).view(d,1))
        for i in acc_list:
            d=len(i)
            acc.append(torch.tensor(i).view(d,1))
        loss=torch.cat(loss,dim=1)
        acc=torch.cat(acc,dim=1)
        # loss_index=loss.argmin(dim=1)+1
        loss_index=acc.argmax(dim=1)+1
        index=0
        sp=myfn.SPath(server_round,None)
        for name in name_list:
            cluster_index=int(loss_index[index].cpu().detach().numpy())
            sp.name="cluster"+str(cluster_index)
            cluster_weight=torch.load(sp.save_model_path())
            clients_proxy[name]["client_data"].proper["cluster_index"]=cluster_index
            for result_index,result in enumerate(test_results):
                if result["name"]==name:
                    test_results[result_index]["test accuracy"]=acc[index,cluster_index-1]
                    test_results[result_index]["test loss"]=loss[index,cluster_index-1]

            sp.name=name
            save_model=sp.save_model_path()
            torch.save(cluster_weight,save_model)
            send_model=sp.send_model_path()
            torch.save(cluster_weight,send_model)
            index+=1

    def tell_client_calculate_eval(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m evaluate parameters \033[0m"+"-"*10)
        sp=myfn.SPath(server_round,None)
        for client_name in clients_proxy:
            sp.name=client_name
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
            clients_proxy[client_name]["client_data"].send2client["path"]=sp.send_model_path()
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]=None
        return clients_proxy
    
    def _get_init_cluster(self,client_name_list,client_name,cluster_num=3):
        cluster_index_dict={}
        cluster_index=1
        for name in client_name_list:
            if cluster_index%cluster_num:
                cluster_index_dict[name]=cluster_index
                cluster_index+=1
            else:
                cluster_index_dict[name]=cluster_index
                cluster_index=1
        return cluster_index_dict[client_name]







        


