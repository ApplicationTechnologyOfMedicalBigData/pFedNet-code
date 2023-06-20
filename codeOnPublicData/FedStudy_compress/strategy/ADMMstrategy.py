import os
import torch
from config import task,model,DEVICE
from typing import Callable, Dict, List, Optional, Tuple, Union
import utils.my_founction as myfn
from utils.my_founction import Clientdata
if task == "classification" and model == "DenseNet3D":
    from network.DenseNet3D import DenseNet3d as Net

if task == "classification" and model == "DenseNet":
    from network.DenseNet import DenseNet as Net

if task == "segmentation" and model == "UNet3D":
    from network.UNet3D import UNet3D as Net

if task == "segmentation" and model == "UNet2D":
    from network.UNet import UNet as Net

net = Net().to(DEVICE)


class ADMMStrategy():

    def global_metric(results):
        total_num = sum([result["samples num"] for result in results])
        global_loss=sum([result["test loss"]*result["samples num"] for result in results])/total_num
        global_acc1=sum([result["test acc1"]*result["samples num"] for result in results])/total_num
        global_acc2=sum([result["test acc2"]*result["samples num"] for result in results])/total_num
        return global_loss,global_acc1,global_acc2

    def global_avg_gradient(fit_results):
        gradients_list=[]
        # if compress:
        for result in fit_results:
            file_path=result["compress"]
            grad_vector_list = myfn.Compress.zip2vector(file_path)
            gradients_list.append(grad_vector_list)
        toltal_sample=0
        gradients=[]
        for index,result in enumerate(fit_results):
            if toltal_sample==0:
                # gradients=gradients_list[index]
                for grad_index,vector in enumerate(gradients_list[index]):
                    gradients.append(vector*result["samples num"])
            else:
                client_grad=gradients_list[index]
                for grad_index,vector in enumerate(gradients_list[index]):
                    gradients[grad_index]=gradients[grad_index]+client_grad[grad_index]*result["samples num"]
            toltal_sample+=result["samples num"]
        for grad_index,vector in enumerate(gradients):
            gradients[grad_index]=vector/toltal_sample
        return gradients

    def tell_clients_calculate_avg_paras(server_round,clients_proxy):
        #init weight
        send_path = "./save_model/server/send/"+"round_"+str(server_round-1)+"/"
        if not os.path.exists(send_path):
                os.makedirs(send_path)
        save_path = "./save_model/server/save/"+"round_"+str(server_round-1)+"/"
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        for client_name in clients_proxy:
            if server_round==1:
                save_model_path=save_path+"model_"+client_name+"_"+task+"_"+model+".pth"
                send_model_path=send_path+"model_"+client_name+"_"+task+"_"+model+".pth"
                clients_proxy[client_name]["client_data"].send2client["path"]=send_model_path
                clients_proxy[client_name]["client_data"].proper["save_path"]=save_model_path
                clients_proxy[client_name]["client_data"].send2client["state"]="init"
                torch.save(net.state_dict(),save_model_path)
                torch.save(net.state_dict(),send_model_path)
            else:
                clients_proxy[client_name]["client_data"].send2client["state"]="avg"
            clients_proxy[client_name]["client"].server_round=server_round
        return clients_proxy

    def tell_client_calculate_personal_paras(server_round:int,clients_proxy:Clientdata,
                            gradients_list:List[torch.tensor],conmunicate_name:List[str])->Clientdata:
        send_path = "./save_model/server/send/"+"round_"+str(server_round)+"/"
        if not os.path.exists(send_path):
            os.makedirs(send_path)
        save_path = "./save_model/server/save/"+"round_"+str(server_round)+"/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for client_name in clients_proxy:
            #send gradients zip
            send_grad_path=send_path+"avg_grad_"+client_name+"_"+task+"_"+model+".zip"
            myfn.Compress.vector2zip(send_grad_path,gradients_list,conmunicate_name["avg"])
            clients_proxy[client_name]["client_data"].send2client["path"]=send_grad_path
            clients_proxy[client_name]["client_data"].send2client["state"]="personal"

            #calculate and save every clint model
            pre_model=torch.load(clients_proxy[client_name]["client_data"].proper["save_path"])
            grad_dict=myfn.VctorAndParas().vector2params(net,gradients_list,conmunicate_name["avg"])
            for name in grad_dict:
                pre_model[name]=pre_model[name]-grad_dict[name]
            now_model_path=save_path+"sub_avg_grad_model_"+client_name+"_"+task+"_"+model+".pth"
            torch.save(pre_model,now_model_path)
            clients_proxy[client_name]["client_data"].proper["avg_sub_model_path"]=now_model_path
        return clients_proxy

    def tell_clients_calculate_eval_metrics(clients_proxy:List[Clientdata],fit_personal_results:List[dict],
                            conmunicate_name:Dict[str,List[str]],server_round:int)->List[Clientdata]:
        name_seq=[]
        grad_path_list=[]
        weight_path_list=[]
        for result in fit_personal_results:
            grad_path_list.append(result["compress"])
            name_seq.append(result["name"])
            weight_path_list.append(clients_proxy[result["name"]]["client_data"].proper["avg_sub_model_path"])
        req_name=conmunicate_name["personal"]
        clients=[clients_proxy[client_name]["client_data"] for client_name in clients_proxy]
        personal_paras=myfn.ClusterClient(grad_path_list,weight_path_list,req_name,clients,name_seq)
        z=personal_paras.get_z()
        send_path= "./save_model/server/send/"+"round_"+str(server_round)+"/"
        save_path= "./save_model/server/save/"+"round_"+str(server_round)+"/"
        for name in z:
            client_model=torch.load(clients_proxy[name]["client_data"].proper["avg_sub_model_path"])
            z_paras=myfn.VctorAndParas().vector2params(net,[z[name]],req_name)
            z_grad_dict={}
            for layer_name in z_paras:
                z_grad_dict[layer_name]=client_model[layer_name]-z_paras[layer_name]
                client_model[layer_name]=z_paras[layer_name]
            save_model_path=save_path+"model_"+name+"_"+task+"_"+model+".pth"
            torch.save(client_model,save_model_path)
            clients_proxy[name]["client_data"].proper["save_path"]=save_model_path

            send_zip_path=send_path+"personal_grad_"+name+"_"+task+"_"+model+".zip"
            z_grad_list=myfn.VctorAndParas().params2vector(z_grad_dict,req_name)
            myfn.Compress.vector2zip(send_zip_path,z_grad_list,req_name)
            clients_proxy[name]["client_data"].send2client["path"]=send_zip_path
            clients_proxy[name]["client_data"].send2client["state"]="eval"

        return clients_proxy

    def get_size_rate(fit_results,path_from="results"):
        global_zip_size=0
        global_raw_size=0
        num=0
        for result in fit_results:
            if path_from == "results":
                file_path=result["compress"]
            elif path_from == "client_proxy":
                file_path=fit_results
            zip_bytes,raw_bytes=myfn.Compress.zipsize(file_path)
            global_zip_size=global_zip_size+zip_bytes/1024
            global_raw_size=global_raw_size+raw_bytes/1024
            num+=1
        return global_zip_size/num,global_raw_size/num