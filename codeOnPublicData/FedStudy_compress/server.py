import os
# set now python path
NOWPATH=os.path.dirname(os.path.realpath(__file__))
os.chdir(NOWPATH)
# set task,model config to function log.server_log
from config import task,model,DEVICE,num_rounds
# 可视化
from torch.utils.tensorboard import SummaryWriter
visual_path = os.path.join("logs", "visualization")
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
writer = SummaryWriter(visual_path)
# build log dir earlier than model Client
from utils import logs 
log = logs.server_log(NOWPATH,task,model)
import warnings
from typing import List,Dict
from utils.my_founction import Clientdata
from utils.my_founction import Compress
# from clients_proxy as CP
from clients_proxy import load_client,get_clients_proxy
import re
from strategy.ADMMstrategy import ADMMStrategy

if task == "classification" and model == "DenseNet3D":
    from network.DenseNet3D import DenseNet3d as Net

if task == "classification" and model == "DenseNet":
    from network.DenseNet import DenseNet as Net

if task == "segmentation" and model == "UNet3D":
    from network.UNet3D import UNet3D as Net

if task == "segmentation" and model == "UNet2D":
    from network.UNet import UNet as Net

warnings.filterwarnings("ignore", category=UserWarning)
net = Net().to(DEVICE)



# def _handle_finished_future_after_fit(
#     future: concurrent.futures.Future,  # type: ignore
#     results: list,
#     failures: list,
# ) -> None:
#     """Convert finished future into either a result or a failure."""
#     # Check if there was an exception
#     failure = future.exception()
#     if failure is not None:
#         failures.append(failure)
#         return
#     # Successfully received a result from a client
#     result = future.result()
#     # Check result status code
#     if future.done():
#         results.append(result)
#         return
##############################################################################
#multy thread is too big, some threads could not run 
# def mul_thread(clients_proxy,run_name,config=[],reback_result=False):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         submitted_fs = {
#             executor.submit(run_fn,client["client"],run_name,client["path"],config)
#             for client in clients_proxy.values()
#         }
#         finished_fs, _ = concurrent.futures.wait(
#             fs=submitted_fs,
#             timeout=None,  # Handled in the respective communication stack
#         )
#         # Gather results
#     results: list = []
#     failures: list = []
#     if reback_result:
#         for future in finished_fs:
#             _handle_finished_future_after_fit(
#                 future=future, results=results, failures=failures
#             )
#     return results, failures
    # return finished_fs
##############################################################################
#redifine to lighten the vol by run in sequence





class Conenect_Clients():
    def __init__(self,clients_proxy:get_clients_proxy,conmunicate_name:List[str],strategy) -> None:
        self.clients_proxy=clients_proxy
        self.conmunicate_name=conmunicate_name
        self.strategy=strategy

        

    def _run_fn(self,client:Clientdata,run_name:str,config:List[str])->Dict[str,str]:
        '''
        return fit or evaluate result one by one client_name -> {"compress": str(zip_path),"name":client_name,"samples_num":int}
        '''
        if run_name=="fit":
            return client["client"].fit(client["client_data"].send2client["path"],config)
        if run_name=="evaluate":
            return client["client"].evaluate(client["client_data"].send2client["path"],config)

    def _mul_thread(self,run_name:str,config:List[str]=[])->List[Dict[str,str]]:
        results: list = []
        for client in self.clients_proxy.values():
            result=self._run_fn(client,run_name,config)
            results.append(result)
        return results

    # active ervery client
    def start_clients(self,compress_config:List[List[str]],log,gama):
        '''
        compress_config: the function that clients comunicate with server, stc,cluaster_stc and other is noncompress function
        1.clients fit avg paras and return the required layer para's gradients to server
        2.server calculates average, return gradients average to clients
        3.clients  fit personal paras and return the required layer para's gradients to server
        4.server calculate personalize layers' paras by admm
        '''
        for server_round in range(1,num_rounds+1):
            
            print("-"*10+"\033[1;33m fit avg parameters \033[0m"+"-"*10)
            self.clients_proxy=self.strategy.tell_clients_calculate_avg_paras(server_round,self.clients_proxy)
            fit_avg_results = self._mul_thread(run_name="fit",config=compress_config)

            gradients_list=self.strategy.global_avg_gradient(fit_avg_results)

            print("-"*10+"\033[1;33m fit personalize parameters \033[0m"+"-"*10)
            self.clients_proxy=self.strategy.tell_client_calculate_personal_paras(server_round,self.clients_proxy,gradients_list,self.conmunicate_name)
            fit_personal_results = self._mul_thread(run_name="fit",config=compress_config)
            
            print("-"*10+"\033[1;33m evaluate all parameters \033[0m"+"-"*10)
            self.clients_proxy=self.strategy.tell_clients_calculate_eval_metrics(self.clients_proxy,fit_personal_results,self.conmunicate_name,server_round)
            test_results = self._mul_thread(run_name="evaluate",config=["all"])

            if task == "classification":
                global_loss,global_acc,global_bacc=self.strategy.global_metric(results=test_results)
                log.logger.info(
                        " server - server_round: {},".format(server_round)+
                        "\033[1;33m global loss: {} \033[0m,".format(global_loss)+
                        "\033[1;33m global accuracy: {} \033[0m".format(global_acc)+
                        "\033[1;33m global bacc: {} \033[0m".format(global_bacc))
                
            if task == "segmentation":
                global_loss,global_acc1,global_acc2=self.strategy.global_metric(results=test_results)

                log.logger.info(
                    " server - server_round: {},".format(server_round)+
                    "\033[1;33m global loss: {} \033[0m,".format(global_loss)+
                    "\033[1;33m global iou: {} \033[0m".format(global_acc1)+
                    "\033[1;33m global label_iou: {} \033[0m".format(global_acc2))

            
            client_send_path="./save_model/client/send/round_"+str(server_round)+"/"
            avg_zip_size,avg_raw_size=Compress.zipsize(client_send_path)

            server_send_path="./save_model/server/send/round_"+str(server_round)+"/"
            global_zip_size,global_raw_size=Compress.zipsize(server_send_path)


            log.logger.info("gama: {}".format(gama)+
                    " ,server size: {} KB".format(global_zip_size)+
                    " ,server compress rate: {}%".format(global_zip_size*100/global_raw_size)+
                    " ,client size: {} KB".format(avg_zip_size)+
                    " ,client compress rate: {}%".format(avg_zip_size*100/avg_raw_size))
            
            save_num=2

            if server_round%save_num==0:
                server_save_dir="./save_model/server/save/"
                # server_save_list=os.listdir(server_save_dir)

                server_send_dir="./save_model/server/send/"
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
        

if __name__=="__main__":
    CLIENTS=load_client()

    # AVGNAME=["conv","norm","classifier"]
    # CONMUNICATE_NAME=["conv","classifier"]
    avgname=["conv"]
    personalize_name=["conv.weight"]
    CONMUNICATE_NAME={"avg":avgname,"personal":personalize_name}



    compress=["other","stc","cluster_stc"]
    fit_configs=[]
    for i in compress:
        fit_config=["all"]
        fit_config.append(i)
        fit_configs.append(fit_config)
    fit_configs=[["all","cluster_stc"]]
    # fit_configs=[["all","stc"],["all","cluster_stc"]]

    #init client_proxy
    # gama_list=[0,50,100,500]
    gama_list=[50]
    for gama_index in range(len(gama_list)):
    # for gama_index in range(1):
        CLIENTSPROXY=get_clients_proxy(CLIENTS,CONMUNICATE_NAME,gama=gama_list[gama_index])
        connect_clients=Conenect_Clients(CLIENTSPROXY,CONMUNICATE_NAME,ADMMStrategy)
        num=0
        for compress_config in fit_configs:
            if gama_index!=0 and ("stc" in compress_config):
                continue
            connect_clients.start_clients(compress_config,log,gama_list[gama_index])
            num+=1
            if num <len(fit_configs):
                log = logs.server_log(NOWPATH,task,model)



