import os
# set now python path
NOWPATH=os.path.dirname(os.path.realpath(__file__))
os.chdir(NOWPATH)
# set task,model config to function log.server_log
from config import task,model,DEVICE
# 可视化
from torch.utils.tensorboard import SummaryWriter
visual_path = os.path.join("logs", "visualization")
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
writer = SummaryWriter(visual_path)
# build log dir earlier than model Client
from utils import logs 
logdir_path = logs.server_log(NOWPATH,model="init")
os.system("cp ./config/dataFederationConfig.csv "+logdir_path)
os.system("cp ./config/UNet3D.json "+logdir_path)
import warnings
import time
from clients_proxy import load_client,get_clients_proxy
from strategy.all_strategy import connect

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

def name_list_fn(net,req_name):
    names=[]
    for name in net.state_dict():
        if req_name in name:
            names.append(name)
    return names

def name_seq_list_fn(net,p):
    long=len(list(net.state_dict().keys()))
    coun=0
    fore_name=[]
    after_name=[]
    weight_dict=net.state_dict()
    for name in weight_dict:
        if weight_dict[name].shape:
            if coun/long<p and coun<long-1:
                fore_name.append(name)
            else:
                after_name.append(name)
        coun+=1
    return fore_name,after_name

def drop_req_name_list(net,per_name_list):
    name_list=list(net.state_dict().keys())
    avg_name_list=[]
    for name in name_list:
        if name not in per_name_list:
            avg_name_list.append(name)
    return avg_name_list

def get_key_names(net,reqname_list):
    net_dict=net.state_dict()
    # net_dict={"decoder1.dec1conv2.weight":1}
    name_list=[]
    for net_name in net_dict:
        get_match=1
        for name in reqname_list:
            if name in net_name:
                flag=1
            else:
                flag=0
            get_match=get_match*flag
        if get_match:
            name_list.append(net_name)
    return name_list
    


'''
CONMUNICATE_NAME must be shape like a dict(str:list(str)): {"avg":["avg_name1","avg_name2",...],"personal":["per_name1","per_name2",...]
'''

if __name__=="__main__":

    data_path="/home/yawei/Documents/liuqinghe/integeration/train_data/unet/lack0/"
    data_path="/home/yawei/Documents/liuqinghe/integeration/train_data/densenet/balance/"
    CLIENTS=load_client(data_path)
    ######################################################################################################################
    #densenet
    # AVGNAME=["conv","norm","classifier"]
    # CONMUNICATE_NAME=["conv","classifier"]
    avgname=["conv","norm"]
    avgname=name_list_fn(net,avgname[0])+name_list_fn(net,avgname[1])
    personalize_name=["classifier"]
    personalize_name=name_list_fn(net,personalize_name[0])
    # avgname,personalize_name=name_seq_list_fn(net,p=0.9)
    ######################################################################################################################
    #unet
    # personalize_name=["conv.weight","conv.bias"]
    # req_perconv_name=["dec","conv"]
    # req_pernorm_name=["dec","norm"]
    # personalize_name=personalize_name+get_key_names(net,req_perconv_name)#+get_key_names(net,req_pernorm_name)
    # avgname=drop_req_name_list(net,personalize_name)

    CONMUNICATE_NAME={"avg":avgname,"personal":personalize_name}

    # warm_start_dict={"type":"personal_model",
    #                  "start_path":"/home/yawei/Documents/berifen/start_model/segmentation/initfeature_8_2d_unet/"}
    
    warm_start_dict=None

    strategy_name={1:"pFedNet",2:"FedAvg",3:"Fedprox",4:"FedAMP",
                   5:"FPFC",6:"FedMGL",7:"FedRoD",8:"Ditto",9:"pFedME",
                   10:"IFCA",11:"SuPerFed",12:"FedPer",13:"FedRep"}
    num=0
    for index in range(1,2):
        strategy_index=13
        log = logs.server_log(logdir_path,task=strategy_name[strategy_index])
        # log = logs.server_log(NOWPATH,task=compress_config)
        server_path="./save_model/server/"
        client_path="./save_model/client/"
        os.system("rm -rf "+ server_path)
        os.system("rm -rf "+ client_path)
        
        #init client_proxy
        CLIENTSPROXY=get_clients_proxy(CLIENTS,CONMUNICATE_NAME,
                                        warm_start_dict=warm_start_dict,strategy_name=strategy_name[strategy_index])
        connect_clients=connect(strategy_name[strategy_index],CLIENTSPROXY,CONMUNICATE_NAME,net)

        connect_clients.start_clients(log)
    # time.sleep(600)
                






























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



