from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata
from client import Client
import re
import os
from config import data_random_split,data_path

def load_client():
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
            clients[index] = Clientdata(name,{},data,{})
            clients[index] = Clientdata(name,{"cov":myfn.cov(data[0])},data,{})
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

def get_clients_proxy(clients,conmunicate_name,server_round=1,gama=10):
    clients_proxy={}
    for client in clients:
        client.send2client["gama"]=gama
        clients_proxy[client.name]={"client":Client(client_name=client.name,client_proper=client.send2client,
                client_data=client.parameters,server_round=server_round
                ,send_name_dict=conmunicate_name),"client_data":client}
    return clients_proxy



