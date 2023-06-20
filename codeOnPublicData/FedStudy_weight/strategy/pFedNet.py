import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata
from config import DEVICE
VP=myfn.VctorAndParas()

class pFedNetStrategy():
    '''
    1.clients fit avg paras and return the required layer para's gradients to server
    2.server calculates average, return gradients average to clients
    3.clients  fit personal paras and return the required layer para's gradients to server
    4.server calculate personalize layers' paras by admm
    '''
    def __init__(self,net) -> None:
        self.net=net

    def tell_clients_calculate_avg_paras(self,server_round,clients_proxy):
        print("-"*10+"\033[1;33m fit avg parameters \033[0m"+"-"*10)
        #init weight
        sp=myfn.SPath(server_round-1,None)
        for client_name in clients_proxy:
            if server_round==1:
                sp.name=client_name
                save_model_path=sp.save_model_path()
                send_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["path"]=send_model_path
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="server_model"
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="part_client_model"
                clients_proxy[client_name]["client_data"].send2client["client_return_name"]="avg"
                clients_proxy[client_name]["client_data"].send2client["server_send_name"]="personal"
                clients_proxy[client_name]["client_data"].send2client["state_grad"]=True
                # clients_proxy[client_name]["client_data"].proper["save_path"]=save_model_path
                torch.save(self.net.state_dict(),save_model_path)
                torch.save(self.net.state_dict(),send_model_path)
            else:
                clients_proxy[client_name]["client_data"].send2client["client_return_name"]="avg"
                clients_proxy[client_name]["client_data"].send2client["server_send_name"]="personal"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="last_client_model"
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="part_client_model"
                clients_proxy[client_name]["client_data"].send2client["state_grad"]=True
            clients_proxy[client_name]["client"].server_round=server_round
        return clients_proxy
    
    def calculate_weight(self,fit_results,server_round):
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
            sp.name=result["name"]
            save_model=sp.save_model_path()
            send_model=sp.send_model_path()
            torch.save(weight_dict,save_model)
            torch.save(weight_dict,send_model)

    def tell_client_calculate_personal_paras(self,server_round:int,clients_proxy:Clientdata,
                            gradients_list:List[torch.tensor],conmunicate_name:List[str])->Clientdata:
        
        print("-"*10+"\033[1;33m fit personalize parameters \033[0m"+"-"*10)
        sp=myfn.SPath(server_round,None)
        for client_name in clients_proxy:
            #send gradients zip
            sp.name=client_name
            send_model=sp.send_model_path()
            clients_proxy[client_name]["client_data"].send2client["path"]=send_model
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="part_server_add_last_client_model"
            clients_proxy[client_name]["client_data"].send2client["server_send_name"]="avg"
            clients_proxy[client_name]["client_data"].send2client["client_return_name"]="personal"
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]="part_gradients_model"
        return clients_proxy

    def tell_clients_calculate_eval_metrics(self,clients_proxy:List[Clientdata],fit_personal_results:List[dict],
                            conmunicate_name:Dict[str,List[str]],server_round:int)->List[Clientdata]:
        
        print("-"*10+"\033[1;33m evaluate all parameters \033[0m"+"-"*10)
        name_seq=[]
        grad_path_list=[]
        weight_path_list=[]
        sp=myfn.SPath(server_round-1,None)
        
        for result in fit_personal_results:
            grad_path_list.append(result["return"])
            name_seq.append(result["name"])
            sp.name=result["name"]
            weight_path_list.append(sp.send_model_path())
        z_dict_list={}
        for name in name_seq:
            z_dict_list[name]=[]
        req_name=conmunicate_name["personal"]

        clients=[clients_proxy[client_name]["client_data"] for client_name in clients_proxy]
        personal_paras=ClusterClient(grad_path_list,weight_path_list,req_name,clients,name_seq)
        z=personal_paras.get_z(epoch=200)

        sp.round=server_round
        for name in z:
            z_paras=VP.vector2params(self.net,z[name],req_name)
            sp.name=name
            send_model_path=sp.send_model_path()
            torch.save(z_paras,send_model_path)

            save_model_path=sp.save_model_path()
            weight_dict=torch.load(save_model_path)
            for layer_name in z_paras:
                weight_dict[layer_name]=z_paras[layer_name]
            torch.save(weight_dict,save_model_path)

            clients_proxy[name]["client_data"].send2client["path"]=send_model_path
            clients_proxy[name]["client_data"].send2client["server_send_type"]="part_server_add_now_client_model"
            clients_proxy[name]["client_data"].send2client["client_return_type"]="client_model"
        return clients_proxy

    

class ClusterClient():
    def __init__(self,grad_path_list:List[str],weight_path_list:List[str],req_name:List[str],
                 clinet_proxy:List[Clientdata],name_seq:List[str],
                 namda=72,eta=0.01) -> None:
        self.grad_tensor=VP.clientpthlist2matrix(grad_path_list,req_name)
        self.weight_tensor=VP.clientpthlist2matrix(weight_path_list,req_name)
        q_matrix=myfn.KNN([],[]).get_q_matrix(self.re_ranke_client(clinet_proxy,name_seq))
        self.q_matrix=q_matrix.T
        self.eye=torch.eye(len(name_seq),device=DEVICE)
        self.g_line,self.g_colume=self.grad_tensor.shape
        self.q_line,self.q_colume=q_matrix.shape
        self.rou=1.0
        self.z=VP.clientpthlist2matrix(weight_path_list,req_name)
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
