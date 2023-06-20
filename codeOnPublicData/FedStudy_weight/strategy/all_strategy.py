from clients_proxy import get_clients_proxy,Conenect_Clients
from typing import List,Dict
from .FedAvg import FedAvgStrategy
from .FedProx import FedProxStrategy
from .FedAMP import FedAMPStrategy
from .FedMGL import FedMGLStrategy
from .FPFC import FPFCStrategy
from .FedRoD import FedRoDStrategy
from .Ditto import DittoStrategy
from .pFedNet import pFedNetStrategy
from .pFedME import pFedMEStrategy
from .IFCA import IFCAStrategy
from .SuPerFed import SuPerFedStrategy
from .FedPer import FedPerStrategy
from .FedRep import FedRepStrategy

class ADMMrun(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_clients_calculate_avg_paras(server_round,self.clients_proxy)
        fit_avg_results = self._mul_thread(run_name="fit")
        gradients_list=self.strategy.global_avg_gradient(fit_avg_results)
        self.clients_proxy=self.strategy.tell_client_calculate_personal_paras(server_round,self.clients_proxy,gradients_list,self.conmunicate_name)
        fit_personal_results = self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_clients_calculate_eval_metrics(self.clients_proxy,fit_personal_results,self.conmunicate_name,server_round)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class pFedNet(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_clients_calculate_avg_paras(server_round,self.clients_proxy)
        fit_avg_results = self._mul_thread(run_name="fit")
        gradients_list=self.strategy.global_avg_gradient(fit_avg_results)
        self.clients_proxy=self.strategy.tell_client_calculate_personal_paras(server_round,self.clients_proxy,gradients_list,self.conmunicate_name)
        fit_personal_results = self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_clients_calculate_eval_metrics(self.clients_proxy,fit_personal_results,self.conmunicate_name,server_round)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results


class FedAvg(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        gradients_list=self.strategy.calculate_weight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy,gradients_list)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class FedProx(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_weight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(self.clients_proxy,server_round)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class FedAMP(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_u(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class FPFC(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_client_caculate_eval(fit_results,server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class FedMGL(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_u(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results
    
class FedRoD(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_global_layers(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_weight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_personal_layers(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_client_caculate_eval(self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results
    
class Ditto(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gloabal_weight(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_weight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_calculate_local_weight(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class pFedNet(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_clients_calculate_avg_paras(server_round,self.clients_proxy)
        fit_avg_results = self._mul_thread(run_name="fit")
        gradients_list=self.strategy.calculate_weight(fit_avg_results,server_round)
        self.clients_proxy=self.strategy.tell_client_calculate_personal_paras(server_round,self.clients_proxy,gradients_list,self.conmunicate_name)
        fit_personal_results = self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_clients_calculate_eval_metrics(self.clients_proxy,fit_personal_results,self.conmunicate_name,server_round)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results
    
class pFedME(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_weight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class IFCA(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        cluster_num=2
        self.clients_proxy=self.strategy.tell_client_calculate_weight(server_round,self.clients_proxy,cluster_num)
        fit_results = self._mul_thread(run_name="fit")
        test_acc=[]
        test_loss=[]
        for cluster_index in range(1,cluster_num+1):
            self.strategy.calculate_weight(fit_results,server_round,self.conmunicate_name,self.clients_proxy,cluster_index)
            self.clients_proxy=self.strategy.tell_client_estimate_loss(server_round,self.clients_proxy,cluster_index)
            cluster_results = self._mul_thread(run_name="fit")
            test_acc.append([result["acc"] for result in cluster_results])
            test_loss.append([result["loss"] for result in cluster_results])
        self.strategy.flush_cluster_index(test_acc,test_loss,self.clients_proxy,server_round,cluster_results)
        self.clients_proxy=self.strategy.tell_client_calculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class SuPerFed(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_FedWeight(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_Globalweight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_calculate_LocalWeight(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_Loaclweight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class FedPer(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_client_calculate_gradients(server_round,self.clients_proxy)
        fit_results = self._mul_thread(run_name="fit")
        gradients_list=self.strategy.calculate_weight(fit_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy,gradients_list)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

class FedRep(Conenect_Clients):
    def __init__(self, clients_proxy: get_clients_proxy, conmunicate_name: List[str], strategy,net) -> None:
        super().__init__(clients_proxy, conmunicate_name, strategy)
        self.strategy=strategy(net)
    def one_round_proceed(self, server_round):
        self.clients_proxy=self.strategy.tell_clients_calculate_personal_paras(server_round,self.clients_proxy)
        self._mul_thread(run_name="fit")
        self.clients_proxy=self.strategy.tell_clients_calculate_avg_paras(server_round,self.clients_proxy)
        fit_avg_results = self._mul_thread(run_name="fit")
        self.strategy.calculate_weight(fit_avg_results,server_round)
        self.clients_proxy=self.strategy.tell_client_caculate_eval(server_round,self.clients_proxy)
        test_results = self._mul_thread(run_name="evaluate")
        return test_results

def connect(requir_strategy,clientsproxy,layer_name,net):
    if requir_strategy=="FedAvg":
        return FedAvg(clientsproxy,layer_name,FedAvgStrategy,net)
    if requir_strategy=="Fedprox":
        return FedProx(clientsproxy,layer_name,FedProxStrategy,net)
    if requir_strategy=="FedAMP":
        return FedAMP(clientsproxy,layer_name,FedAMPStrategy,net)
    if requir_strategy=="FPFC":
        return FPFC(clientsproxy,layer_name,FPFCStrategy,net)
    if requir_strategy=="FedMGL":
        return FedMGL(clientsproxy,layer_name,FedMGLStrategy,net)
    if requir_strategy=="FedRoD":
        return FedRoD(clientsproxy,layer_name,FedRoDStrategy,net)
    if requir_strategy=="Ditto":
        return Ditto(clientsproxy,layer_name,DittoStrategy,net)
    if requir_strategy=="pFedNet":
        return pFedNet(clientsproxy,layer_name,pFedNetStrategy,net)
    if requir_strategy=="pFedME":
        return pFedME(clientsproxy,layer_name,pFedMEStrategy,net)
    if requir_strategy=="IFCA":
        return IFCA(clientsproxy,layer_name,IFCAStrategy,net)
    if requir_strategy=="SuPerFed":
        return SuPerFed(clientsproxy,layer_name,SuPerFedStrategy,net)
    if requir_strategy=="FedPer":
        return FedPer(clientsproxy,layer_name,FedPerStrategy,net)
    if requir_strategy=="FedRep":
        return FedRep(clientsproxy,layer_name,FedRepStrategy,net)








