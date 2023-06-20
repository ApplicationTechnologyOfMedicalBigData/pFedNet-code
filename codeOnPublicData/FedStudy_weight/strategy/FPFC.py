import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from config import DEVICE,lr
VP=myfn.VctorAndParas()


class CalculateParams():
    def __init__(self,v,files_path_list,net,server_round,
                 rou=1,sigma=1,namda=1,a=10) -> None:
        
        self.files_path_list=files_path_list
        self.name_list=VP.all_name_list(net)
        self.net=net
        self.rou=rou
        self.sigma=sigma
        self.namda=namda
        self.a=a
        self.m,self.d=self.parameters_size()
        if server_round==1 :
            self.v=self.init_v()
        else:
            self.v=v
        self.weight_matrix=self._get_vector_list()
        self.sita=self._calculate_sita()

    def parameters_size(self):
        weight_dict=torch.load(self.files_path_list[0])
        weight_vector=VP.params2vector(weight_dict,self.name_list)
        d,_=weight_vector.shape
        m=len(self.files_path_list)
        return int(m),int(d)

    def _get_vector_list(self):
        weight_vector_list=[]
        for file_path in self.files_path_list:
            weight_dict=torch.load(file_path)
            weight_vector=VP.params2vector(weight_dict,self.name_list)
            weight_vector_list.append(weight_vector)
        weight_matrix=torch.cat(weight_vector_list,dim=1)
        return weight_matrix
    
    def loc2index(self,row,col):
        m=len(self.files_path_list)
        index=0
        index_matrix=torch.zeros((m,m),device=DEVICE)
        for i in range(m):
            for j in range(m):
                if i<j:
                    index_matrix[i,j]=index
                    index+=1
                if i==j:
                    index_matrix[i,j]=m*m
        index_matrix=index_matrix+index_matrix.T
        return index_matrix[row,col]
    
    def index2loc(self,index):
        m=len(self.files_path_list)
        matrix_index=0
        for i in range(m):
            for j in range(m):
                if i<j:
                    if index==matrix_index:
                        return i,j
                    matrix_index+=1
        return None
    
    def init_v(self):
        return torch.zeros((self.m,self.m,self.d),device=DEVICE)
    
    def _calculate_deta(self):
        deta=torch.zeros((self.m,self.m,self.d),device=DEVICE)
        for i in range(self.m):
            for j in range(i+1,self.m):
                single_deta=self.weight_matrix[:,i]-self.weight_matrix[:,j]+self.v[i,j,:]/self.rou
                deta[i,j,:]=single_deta
        return deta
    
    def _coefficient(self,regular_deta):
        if regular_deta<=self.sigma+self.namda/self.rou:
            return self.sigma*self.rou/(self.namda+self.sigma*self.rou)
        elif regular_deta>self.sigma+self.namda/self.rou and regular_deta<=self.namda+self.namda*self.rou:
            return 1-self.namda/(self.rou*regular_deta)
        elif regular_deta>self.namda+self.namda*self.rou and regular_deta<=self.a*self.namda:
            up_coeff=max(0,(1-self.a*self.namda/(self.a-1)*self.rou*regular_deta))
            sub_coeff=1-1/int((self.a-1)*self.rou)
            return up_coeff/sub_coeff
        elif regular_deta>self.a*self.namda:
            return 1.0
    
    def _calculate_sita(self):
        deta=self._calculate_deta()
        sita=torch.zeros((self.m,self.m,self.d),device=DEVICE)
        for i in range(self.m):
            for j in range(i+1,self.m):
                regular_deta=torch.norm(deta[i,j,:],p=1)
                regular_deta=regular_deta/self.d
                coeffic=self._coefficient(regular_deta)
                sita[i,j,:]=coeffic*deta[i,j,:]
        return sita
    
    def calculate_results(self,name_seq):
        for i in range(self.m):
            for j in range(self.m):
                self.v[i,j,:]=self.v[i,j,:]+self.rou*(self.weight_matrix[:,j]-self.weight_matrix[:,j]-self.sita[i,j,:])
                self.v[j,i,:]=-1*self.v[i,j,:]
                self.sita[j,i,:]=-1*self.sita[i,j,:]
        paras=torch.zeros((self.d,self.m),device=DEVICE)
        paras_dict={}
        
        for i in range(self.m):
            for j in range(self.m):
                paras[:,i]=paras[:,i]+self.weight_matrix[:,j]+self.sita[i,j,:]-self.v[i,j,:]/self.rou
            paras[:,i]=lr*(self.weight_matrix[:,i]-paras[:,i]/self.m)*self.rou
            paras_dict[name_seq[i]]=VP.vector2params(self.net,paras[:,i],self.name_list)
        return paras_dict,self.v
        


class FPFCStrategy():
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
                clients_proxy[client_name]["client_data"].send2client["server_model_path"]=send_model_path
                torch.save(self.net.state_dict(),save_model_path)
                torch.save(self.net.state_dict(),send_model_path)
            else:
                sp.name=client_name
                send_model_path=sp.send_model_path()
                clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_model"
                clients_proxy[client_name]["client_data"].send2client["server_send_type"]="last_client_model"
                clients_proxy[client_name]["client_data"].send2client["server_model_path"]=send_model_path
            clients_proxy[client_name]["client"].server_round=server_round
        return clients_proxy
            
    def tell_client_caculate_eval(self,fit_results,server_round,clients_proxy):
        sp=myfn.SPath(None,"server_para_v")
        
        if server_round==1:
            name_seq=[result["name"] for result in fit_results]
            files_path_list=[result["return"] for result in fit_results]
            v=CalculateParams(None,files_path_list,self.net,server_round).init_v()
        else:
            files_path_list=[]
            sp.round=server_round-1
            last_save_v_path=sp.save_model_path()
            
            params_dict=torch.load(last_save_v_path)
            v=params_dict["v"]
            name_seq=params_dict["name_seq"]
            for name in name_seq:
                for result in fit_results:
                    if name==result["name"]:
                        files_path_list.append(result["return"])
                        break

        server_parameters=CalculateParams(v,files_path_list,self.net,server_round)
        para_dict,v=server_parameters.calculate_results(name_seq)
        sp.round=server_round
        now_save_v_path=sp.save_model_path()
        torch.save({"v":v,"name_seq":name_seq},now_save_v_path)  

        for client_name in clients_proxy:
            sp.name=client_name
            send_path=sp.send_model_path()
            torch.save(para_dict[client_name],send_path)
            clients_proxy[client_name]["client_data"].send2client["server_model_path"]=send_path
            clients_proxy[client_name]["client_data"].send2client["server_send_type"]="now_client_model"
            clients_proxy[client_name]["client_data"].send2client["client_return_type"]="client_sub_paras_model"
        return clients_proxy
        
        



            









