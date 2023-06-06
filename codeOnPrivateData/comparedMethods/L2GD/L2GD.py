import numpy as np 
from settingsOfL2GD import FormulationSettings as FormulationSettingsOfL2GD
import os, sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from settings import DataFederation as DataFederationOfL2GD, FederatedLearningSettings as FederatedLearningSettingsOfL2GD
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from tools import ModelLogger

class Optimizer(object):
    """
    The optimizer to solve local models (for clients), NO global models (for servers).
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfL2GD))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfL2GD))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfL2GD))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__X = np.zeros((self.__d, self.__N)) # local models
        self.__lambda = self.__formulationSettings.getLambda()
        pass
    
    def getPersonalizedModel(self):
        return self.__X
    
    def updateModelAtClient(self, givenEta, givenPOfXi):
        Grads = self.__specificModelObj.computePersonalizedGradients()
        self.__X = self.__X - givenEta / (self.__N * (1-givenPOfXi)) * Grads
        return self.__X
    
    def updateModelAtServer(self, givenEta, givenPOfXi):
        xBar = np.mean(self.__X, axis=1)
        balanceWeight = givenEta*self.__lambda / (self.__N * givenPOfXi)
        self.__X = (1 - balanceWeight) * self.__X + balanceWeight * np.einsum('i,j->ij', xBar, np.ones(self.__N))
        return self.__X
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfL2GD))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfL2GD))
        self.__name = 'L2GD'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings =  FormulationSettingsOfL2GD(givenDatasetObj=self.__dataFederation.getDatasetObj(), givenLambda = 0.1) 
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
            pOfXi = np.random.random()
            stepSize = 1e-3/(np.sqrt(i+1))
            opt.updateModelAtClient(givenEta=stepSize, givenPOfXi=pOfXi)
            opt.updateModelAtServer(givenEta=stepSize, givenPOfXi=pOfXi)
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
        federatedLearningSettings = FederatedLearningSettingsOfL2GD()
        dataFederation = DataFederationOfL2GD(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
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








