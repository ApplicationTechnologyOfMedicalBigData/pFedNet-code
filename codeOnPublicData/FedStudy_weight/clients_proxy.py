import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata
# from .utils import my_founction as myfn
# from .utils.my_founction import Clientdata
from client import Client
import re
import os
from config import data_random_split,num_rounds,task
from abc import ABC, abstractmethod
import time
def load_client(data_path):
    # get every client data
    if data_random_split:
        from utils.dataset import load_random_split_data as load_data
        #do not read every client
        print("-"*10+"\033[2;31m  load data \033[0m"+"-"*10)
        client_dir=[]
        for i in os.listdir(data_path):
            if not re.findall("csv",i):
                client_dir.append(i)
        clients=[Clientdata(None,{},[],{})]*len(client_dir)
        for index,name in enumerate(client_dir):
            client_path=data_path+"/"+name
            data=load_data(client_path)
            # clients[index] = Clientdata(name,{},data,{})
            clients[index] = Clientdata(name,{"cov":myfn.cov(data[0],task=task)},data,{})
        clients=_get_KNN(clients)
        for client in clients:
            print(client.name+" : ",client.proper["near clients"])
            client.proper.pop("cov")
        print("-"*10+"\033[1;31m load data done \033[0m"+"-"*10)
        return clients
    
# get near k client
def _get_KNN(clients_data:List[Clientdata],k:int = 3)->dict:
    print("-"*10 + "get near KNN" + "-"*10)
    long=len(clients_data)
    name=[client.name for client in clients_data]
    cov_list=[client.proper["cov"] for client in clients_data]
    distance_metrix=myfn.KNN(name,cov_list)
    index_list=distance_metrix.get_KNN(k)
    for i in range(long):
        clients_data[i].proper["near clients"]=[clients_data[client_num].name for client_num in index_list[i]]
    return clients_data

def get_clients_proxy(clients,conmunicate_name:List[str]=None,server_round=1,warm_start_dict={},strategy_name=None):
    clients_proxy={}
    for client in clients:
        client.send2client["client_return_name"]="all"
        client.send2client["client_return_type"]="part_client_zip"
        client.send2client["server_send_name"]="all"
        client.send2client["server_send_type"]="part_server_zip"
        client.send2client["Fedloss"]="normal"
        client.send2client["state_grad"]=False
        client.send2client["server_model_path"]=None
        client.proper["warm_start_dict"]=warm_start_dict
        client.proper["fit_count"]=0
        client.proper["strategy_name"]=strategy_name
        clients_proxy[client.name]={"client":Client(client_name=client.name,client_proper=client.send2client,
                client_data=client.parameters,server_round=server_round
                ,send_name_dict=conmunicate_name),"client_data":client}
    return clients_proxy


class Conenect_Clients(ABC):
    def __init__(self,clients_proxy:get_clients_proxy,conmunicate_name:List[str],strategy) -> None:
        self.clients_proxy=clients_proxy
        self.conmunicate_name=conmunicate_name
        self.strategy=strategy
                

    def _run_fn(self,client:Clientdata,run_name:str,config:List[str])->Dict[str,str]:
        '''
        return fit or evaluate result one by one client_name -> {"compress": str(zip_path),"name":client_name,"samples_num":int}
        '''
        self.strategy_name=client["client_data"].proper["strategy_name"]

        if run_name=="fit":
            if client["client_data"].proper["fit_count"]==0 and client["client_data"].proper["warm_start_dict"]:
                warm_start_dict=client["client_data"].proper["warm_start_dict"]
                if warm_start_dict["type"]=="one_model":
                    client["client_data"].send2client["path"]=warm_start_dict["start_path"]
                elif warm_start_dict["type"]=="personal_model":
                    clients_path_list=[warm_start_dict["start_path"]+path for path in os.listdir(warm_start_dict["start_path"])]
                    for path in clients_path_list:
                        match_str=".*({}).*".format(client["client_data"].name)
                        if re.findall(match_str,path):
                            client["client_data"].send2client["path"]=path


            result=client["client"].fit(client["client_data"].send2client["path"],config)
            client["client_data"].proper["fit_count"]=client["client_data"].proper["fit_count"]+1
        if run_name=="evaluate":
            result=client["client"].evaluate(client["client_data"].send2client["path"],config)
        return result
        
        
    def _mul_thread(self,run_name:str,config:List[str]=["all"])->List[Dict[str,str]]:
        results: list = []
        for client in self.clients_proxy.values():
            result=self._run_fn(client,run_name,config)
            results.append(result)
        return results
    def global_metric(self,results):
        total_num = sum([result["samples num"] for result in results])
        global_loss=sum([result["test_loss"]*result["samples num"] for result in results])/total_num
        global_acc1=sum([result["test_acc1"]*result["samples num"] for result in results])/total_num
        global_acc2=sum([result["test_acc2"]*result["samples num"] for result in results])/total_num
        return global_loss,global_acc1,global_acc2
    @abstractmethod
    def one_round_proceed(self,server_round,compress_config):
        '''
        compress_config: the function that clients comunicate with server, stc,cluaster_stc and other is noncompress function
        return test_results
        '''
    # active ervery client
    def start_clients(self,log):
        for server_round in range(1,num_rounds+1):
            test_results=self.one_round_proceed(server_round)
            global_loss,global_acc1,global_acc2=self.global_metric(results=test_results)
            save_num=5

            if server_round%save_num==0:
                server_save_dir="./save_model/server/save/"
                # server_save_list=os.listdir(server_save_dir)

                server_send_dir="./save_model/server/send/"
                if server_round==num_rounds:
                    useful_model_path="./useful_model/"+self.strategy_name
                    if not os.path.exists(useful_model_path):
                        os.makedirs(useful_model_path)
                    os.system("cp -r "+server_save_dir+"round_"+str(server_round)+"/* "+useful_model_path)
                server_send_list=os.listdir(server_send_dir)

                client_send_dir="./save_model/client/send/"
                # client_send_list=os.listdir(client_send_dir)

                client_save_dir="./save_model/client/save/"
                # client_save_list=os.listdir(client_save_dir)

                server_num_list=[]
                for round_name in server_send_list:
                    server_num=re.findall("round_([\d]+)",round_name)
                    server_num=int(server_num[0])
                    if server_num<server_round:
                        server_num_list.append(round_name)

                server_save_path_list=[server_save_dir+path for path in server_num_list]
                server_send_path_list=[server_send_dir+path for path in server_num_list]
                client_send_path_list=[client_send_dir+path for path in server_num_list]
                client_save_path_list=[client_save_dir+path for path in server_num_list]
                # client_path="./save_model/client/"
                for i in range(len(server_save_path_list)):
                    os.system("rm -rf "+ server_save_path_list[i])
                    os.system("rm -rf "+ server_send_path_list[i])
                    os.system("rm -rf "+ client_send_path_list[i])
                    os.system("rm -rf "+ client_save_path_list[i])
                    # os.system("rm -rf "+ client_path)
            
            if task == "segmentation":
                log.logger.info(
                    " server - server_round: {},".format(server_round)+
                    "\033[1;33m global loss: {} \033[0m,".format(global_loss)+
                    "\033[1;33m global iou: {} \033[0m".format(global_acc1)+
                    "\033[1;33m global label_iou: {} \033[0m".format(global_acc2))

            if task == "classification":
                log.logger.info(
                    " server - server_round: {},".format(server_round)+
                    "\033[1;33m global loss: {} \033[0m,".format(global_loss)+
                    "\033[1;33m global accuracy: {} \033[0m".format(global_acc1)+
                    "\033[1;33m global bacc: {} \033[0m".format(global_acc2))
                
            # if server_round%100 == 0:
            #     time.sleep(300)




