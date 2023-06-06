"""
Ref. Ditto: Fair and Robust Federated Learning Through Personalization.
"""
import numpy as np
from settingsOfDitto import FormulationSettings as FormulationSettingsOfDitto
import os, sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from settings import DataFederation as DataFederationOfDitto, FederatedLearningSettings as FederatedLearningSettingsOfDitto
from tools import ModelLogger

class Optimizer(object):
    """
    The optimizer to solve local models (for clients), and global models (for servers).
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfDitto))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfDitto))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfDitto))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()  
        self.__eta, self.__lambdav = 1e-3, 0.5
        self.__omega = np.zeros(self.__d) # global model
        self.__V = np.zeros((self.__d, self.__N)) # personalized models  
        self.__Delta = np.zeros((self.__d, self.__N)) # difference matrix  
        pass
    
    def getPersonalizedModel(self):
        return self.__V
    
    def __updateV(self):
        Omega = np.einsum('i,j->ij', self.__omega, np.ones(self.__N))
        for iter in range(10): # s = 10
            Grads = self.__specificModelObj.fetchPersonalziedGradient(givenPersonalizedModel = self.__V)
            self.__V = self.__V - self.__eta / np.sqrt(1+iter) * (Grads + self.__lambdav*(self.__V - Omega))
        return self.__V
    
    def updateModelAtClient(self):
        numOfIters = 10
        Omega = np.einsum('i,j->ij', self.__omega, np.ones(self.__N))
        for iter in range(numOfIters):
            Grads = self.__specificModelObj.fetchPersonalziedGradient(givenPersonalizedModel = Omega)
            Omega = Omega - 0.1/np.sqrt(iter+1) * Grads
        self.__Delta = Omega - np.einsum('i,j->ij', self.__omega, np.ones(self.__N))
        self.__updateV()
        return Omega
    
    def updateModelAtServer(self):
        self.__omega = np.sum(self.__Delta + np.einsum('i,j->ij', self.__omega, np.ones(self.__N)), axis = 1) / self.__N
        return self.__omega
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfDitto))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfDitto))
        self.__name = 'Ditto'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfDitto(givenDatasetObj=self.__dataFederation.getDatasetObj(), givenLambda=0.1) 
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__specificModelObj = self.__initPersonalizedModel(givenName = 'federated logistic regression')
        pass
    
    def __initPersonalizedModel(self, givenName):
        if givenName == 'federated logistic regression':
            specificModel = FederatedLogisticRegression(givenPersonalizedModels = np.zeros((self.__d, self.__N)), givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), 
                                                        givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), 
                                                        givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
        return specificModel
    
    def execute(self):
        opt = Optimizer(givenDataFederation=self.__dataFederation, givenFormulationSettings=self.__formulationSettings, givenFederatedLearningSettings=self.__federatedLearningSettings, givenSpecificModelObj = self.__specificModelObj)
        for i in range(self.__federatedLearningSettings.getNumOfIterations()):
            self.__specificModelObj.appendRecords()
            # update on clients.
            opt.updateModelAtClient()
             # update on server.
            opt.updateModelAtServer()
            self.__specificModelObj.updatePersonalziedModels(givenXtn = opt.getPersonalizedModel())
        return 
    
    def logAccuracy(self, givenTypeList = ['training', 'test']):
        if 'test' in givenTypeList:
            ModelLogger(givenSpecificModel = self.__specificModelObj).logTestAccuracy(givenDatasetName = self.__dataFederation.getDatasetObj().getName(), 
                                                                                givenNumOfClients = self.__federatedLearningSettings.getNumOfClients(), 
                                                                                givenMethod = self.__name)
        if 'training' in givenTypeList:
            ModelLogger(givenSpecificModel = self.__specificModelObj).logTrainingAccuracy(givenDatasetName = self.__dataFederation.getDatasetObj().getName(), 
                                                                                givenNumOfClients = self.__federatedLearningSettings.getNumOfClients(), 
                                                                                givenMethod = self.__name)
        pass
    
    def logLoss(self, givenTypeList = ['training', 'test']):
        if 'test' in givenTypeList:
            ModelLogger(givenSpecificModel = self.__specificModelObj).logTestLoss(givenDatasetName = self.__dataFederation.getDatasetObj().getName(), 
                                                                                givenNumOfClients = self.__federatedLearningSettings.getNumOfClients(), 
                                                                                givenMethod = self.__name)
        if 'training' in givenTypeList:
            ModelLogger(givenSpecificModel = self.__specificModelObj).logTrainingLoss(givenDatasetName = self.__dataFederation.getDatasetObj().getName(), 
                                                                                givenNumOfClients = self.__federatedLearningSettings.getNumOfClients(), 
                                                                                givenMethod = self.__name)
        pass
    
    def logFinalModel(self):
        ModelLogger(givenSpecificModel = self.__specificModelObj).logFinalModel(givenDatasetName = self.__dataFederation.getDatasetObj().getName(), 
                                                                                givenNumOfClients = self.__federatedLearningSettings.getNumOfClients(),
                                                                                givenMethod = self.__name)
        pass
        
    def getName(self):
        return self.__name
    pass

class Evaluater(object):
    def __init__(self, givenDatasetObj):
        self.__datasetObj = givenDatasetObj 
        pass
        
    def execute(self):
        federatedLearningSettings = FederatedLearningSettingsOfDitto()
        dataFederation = DataFederationOfDitto(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
        pcc = PersonalizedFederatedLearning(givenDataFederation=dataFederation,  givenFederatedLearningSettings=federatedLearningSettings)
        pcc.execute()
        # save test and training accuracy
        pcc.logAccuracy()
        # save test and training loss
        pcc.logLoss()
        # save final model
        pcc.logFinalModel()
        pass



class Test(object):
    def testPersonalizedModel(self):
        #datasetObj = ToyDataOfConvexCluster(givenNumberOfSamples=100, givenNumberOfFeatures=6)
        #datasetObj = CovtypeOfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/covtype.txt')
        #datasetObj = Ijcnn1OfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/ijcnn1.txt')
        #datasetObj = BreastCancerOfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/breastCancer.txt')
        datasetObj = LngxbhbcdexzlhzcxfxycDataset(givenFn='/home/yawei/data/medicalDatasets/real/classification/老年冠心病合并肠道恶性肿瘤患者的出血风险预测.csv')
        #datasetObj = IrisOfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/iris.txt')
        #datasetObj = HalfMoonsDataset(givenFn = '/home/yawei/data/others/clustering/halfMoons.csv')
        Evaluater(givenDatasetObj=datasetObj).execute()
        pass
    pass


if __name__ == '__main__':
    np.random.seed(0)
    Test().testPersonalizedModel()







































