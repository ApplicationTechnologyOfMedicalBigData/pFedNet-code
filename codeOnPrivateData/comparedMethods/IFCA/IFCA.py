"""
Paper: An efficient framework for clustered federated learning. Avishek Ghosh et al.
"""

import numpy as np 
from settingsOfIFCA import FormulationSettings as FormulationSettingsOfIFCA
import sys, os
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from settings import DataFederation as DataFederationOfIFCA, FederatedLearningSettings as FederatedLearningSettingsOfIFCA
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression, StandardLogisticRegression
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from tools import ModelLogger

class Optimizer(object):
    """
    The optimizer to solve local models (for clients), NO global models (for servers).
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfIFCA))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfIFCA))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfIFCA))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__Theta = np.zeros((self.__d, self.__formulationSettings.getNumOfClusters())) # representive models for all clusters
        self.__clusterIdForClients = self.__findJHat()
        self.__X = np.zeros((self.__d, self.__N)) # personalized models
        self.__aveGrads = np.zeros((self.__d, self.__N)) 
        pass
    
    def getPersonalizedModel(self):
        return self.__X

    def __updatePersonalizedModel(self):
        for clienti in range(self.__N):
            self.__X[:,clienti] = self.__Theta[:,self.__clusterIdForClients[clienti]]
        return self.__X
    
    def __findJHat(self):
        realClusterIdList = []
        allSampleMat = self.__dataFederation.getDatasetObj().getSampleMat()
        allLabelVec = self.__dataFederation.getDatasetObj().getLabelVec()
        sampleIdList = self.__dataFederation.getSampleIdListOfClient()
        for clienti in range(self.__N):
            lossWrtClusterList = []
            for clusterId in range(self.__formulationSettings.getNumOfClusters()):
                localLoss = StandardLogisticRegression(givenModel = self.__Theta[:, clusterId], givenSampleMat = allSampleMat[sampleIdList[clienti],:], givenLabelVec = allLabelVec[sampleIdList[clienti]]).getLoss()
                lossWrtClusterList.append(localLoss)
            realClusterId = lossWrtClusterList.index(min(lossWrtClusterList))
            realClusterIdList.append(realClusterId)
        self.__clusterIdForClients = realClusterIdList
        return self.__clusterIdForClients
    
    def updateModelAtClient(self):
        self.__findJHat()
        self.__aveGrads = self.__specificModelObj.computePersonalizedGradients()
        return self.__aveGrads
    
    def updateModelAtServer(self, givenEta):
        for clusterId in range(self.__formulationSettings.getNumOfClusters()):
            if clusterId not in self.__clusterIdForClients:
                continue
            grads = self.__aveGrads[:, np.array(self.__clusterIdForClients) == clusterId]
            self.__Theta[:,clusterId] = self.__Theta[:,clusterId] - givenEta * np.sum(grads, axis=1)
        self.__updatePersonalizedModel()
        return 
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfIFCA))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfIFCA))
        self.__name = 'IFCA'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfIFCA(givenDatasetObj=self.__dataFederation.getDatasetObj(), givenNumOfClusters=3) # 3 clusters
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
    
    def executeBySGD(self):
        opt = Optimizer(givenDataFederation=self.__dataFederation, givenFormulationSettings=self.__formulationSettings, givenFederatedLearningSettings = self.__federatedLearningSettings, givenSpecificModelObj = self.__specificModelObj)
        for i in range(self.__federatedLearningSettings.getNumOfIterations()):
            self.__specificModelObj.appendRecords()
            stepSize = 1e-3/(np.sqrt(i+1))
            opt.updateModelAtClient()
            opt.updateModelAtServer(givenEta=stepSize)
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
        federatedLearningSettings = FederatedLearningSettingsOfIFCA()
        dataFederation = DataFederationOfIFCA(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
        pcc = PersonalizedFederatedLearning(givenDataFederation=dataFederation,  givenFederatedLearningSettings=federatedLearningSettings)
        pcc.executeBySGD()
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






if __name__ == "__main__":
    np.random.seed(0)
    Test().testPersonalizedModel()











