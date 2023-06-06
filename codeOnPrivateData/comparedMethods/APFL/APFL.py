import numpy as np 
from settingsOfAPFL import FormulationSettings as FormulationSettingsOfAPFL
import os, sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from settings import DataFederation as DataFederationOfAPFL, FederatedLearningSettings as FederatedLearningSettingsOfAPFL
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from tools import ModelLogger

class Optimizer(object):
    """
    The optimizer to solve local models (for clients), NO global models (for servers).
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfAPFL))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfAPFL))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfAPFL))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__W, self.__V, self.__VBar = np.zeros((self.__d, self.__N)), np.zeros((self.__d, self.__N)), np.zeros((self.__d, self.__N))
        self.__w = np.zeros(self.__d) # global model
        self.__synchronStep = 1
        self.__alpha = 0.05
        pass
    
    def getGlobalModel(self):
        return self.__w
    
    def getPersonalizedModel(self):
        return self.__W
    
    def fetchPersonalizedModel(self):
        pModel = self.__alpha * self.__V + (1-self.__alpha) * np.einsum('i,j->ij', self.__w, np.ones(self.__N))
        return pModel
    
    def __updateW(self, givenEta):
        flr = FederatedLogisticRegression(givenPersonalizedModels = self.__W, givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), 
                                        givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
        wGrad = flr.computePersonalizedGradients()
        self.__W = self.__W - givenEta * wGrad
        return self.__W
    
    def __updateV(self, givenEta):
        flr = FederatedLogisticRegression(givenPersonalizedModels = self.__VBar, givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), 
                                        givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
        vGrad = flr.computePersonalizedGradients()
        self.__V = self.__V - givenEta * vGrad
        return self.__V
    
    def updateModelAtClient(self, givenEta):
        for stepi in range(self.__synchronStep):
            self.__updateW(givenEta)
            self.__updateV(givenEta)
            self.__VBar = self.__alpha * self.__V + (1-self.__alpha)*self.__W
        pass
    
    def updateModelAtServer(self):
        self.__w = np.mean(self.__W, axis=1)
        self.__W = np.einsum('i,j->ij', self.__w, np.ones(self.__N))
        pass
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfAPFL))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfAPFL))
        self.__name = 'APFL'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfAPFL(givenDatasetObj=self.__dataFederation.getDatasetObj()) 
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
            opt.updateModelAtClient(givenEta=stepSize)
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
        federatedLearningSettings = FederatedLearningSettingsOfAPFL()
        dataFederation = DataFederationOfAPFL(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
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
        datasetObj = BreastCancerOfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/breastCancer.txt')
        #datasetObj = LngxbhbcdexzlhzcxfxycDataset(givenFn='/home/yawei/data/medicalDatasets/real/classification/老年冠心病合并肠道恶性肿瘤患者的出血风险预测.csv')
        #datasetObj = IrisOfLibsvm(givenFn='/home/yawei/data/libsvmDatasets/classification/iris.txt')
        #datasetObj = HalfMoonsDataset(givenFn = '/home/yawei/data/others/clustering/halfMoons.csv')
        Evaluater(givenDatasetObj=datasetObj).execute()
    pass






if __name__ == "__main__":
    np.random.seed(0)
    Test().testPersonalizedModel()







