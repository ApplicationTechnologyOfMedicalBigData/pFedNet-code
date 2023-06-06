"""
Ref. Personalized Federated Learning with Moreau Envelopes.
"""
import numpy as np 
from settingsOfpFedMe import FormulationSettings as FormulationSettingsOfpFedMe
import os, sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from settings import DataFederation as DataFederationOfpFedMe, FederatedLearningSettings as FederatedLearningSettingsOfpFedMe
from tools import ModelLogger

class Optimizer(object):
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfpFedMe))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfpFedMe))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfpFedMe))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()  
        self.__eta, self.__beta = 1e-3, 0.5
        self.__lambdav = self.__formulationSettings.getLambda()
        self.__omega = np.zeros(self.__d) # global model
        self.__Omega = np.zeros((self.__d, self.__N)) # local models   
        pass
    
    def getPersonalizedModel(self):
        return self.__Omega

    def __computeThetaTildeOnClient(self):
        """
        Solve Eq. (7).
        
        *givenOmega*: d by N
        """
        ThetaTildeOfClients = np.zeros((self.__d, self.__N))
        numOfInnerIters = 10
        for iter in range(numOfInnerIters):
            flr = FederatedLogisticRegression(givenPersonalizedModels = ThetaTildeOfClients, givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), 
                                        givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), 
                                        givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
            Grads = flr.computePersonalizedGradients() + self.__lambdav*(ThetaTildeOfClients - self.__Omega)
            ThetaTildeOfClients = ThetaTildeOfClients - 1e-3/np.sqrt(iter+1) * Grads
        return ThetaTildeOfClients
    
    def updateModelAtClient(self):
        R = 3
        self.__Omega = np.einsum('i,j->ij', self.__omega, np.ones(self.__N))
        for iter in range(R):
            ThetaTildeOfClients = self.__computeThetaTildeOnClient()
            self.__Omega = self.__Omega - self.__eta / (iter + 1) * self.__lambdav * (self.__Omega - ThetaTildeOfClients)
        return self.__Omega
    
    def updateModelAtServer(self):
        self.__omega = (1-self.__beta) * self.__omega + self.__beta / self.__N * np.sum(self.__Omega, axis = 1)
        return self.__omega
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfpFedMe))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfpFedMe))
        self.__name = 'pFedMe'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfpFedMe(givenDatasetObj=self.__dataFederation.getDatasetObj(), givenLambda=0.1)
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
        opt = Optimizer(givenDataFederation=self.__dataFederation, givenFormulationSettings=self.__formulationSettings, givenFederatedLearningSettings = self.__federatedLearningSettings, givenSpecificModelObj = self.__specificModelObj)
        for i in range(self.__federatedLearningSettings.getNumOfIterations()):
            self.__specificModelObj.appendRecords()
            opt.updateModelAtClient()
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
        federatedLearningSettings = FederatedLearningSettingsOfpFedMe()
        dataFederation = DataFederationOfpFedMe(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
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






if __name__ == "__main__":
    np.random.seed(0)
    Test().testPersonalizedModel()





