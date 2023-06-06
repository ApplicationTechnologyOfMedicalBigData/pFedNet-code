import numpy as np 
from settingsOfFedAvg import FormulationSettings as FormulationSettingsOfFedAvg
import sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm, IITnbswmbbfxycDataset
from settings import DataFederation as DataFederationOfFedAvg, FederatedLearningSettings as FederatedLearningSettingsOfFedAvg
from tools import ModelLogger

class Optimizer(object):
    """
    The cvx optimizer to solve the global model. 
    
    min_{W} F(W)
    """
    def __init__(self, givenDataFederation, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfFedAvg))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfFedAvg))
        self.__dataFederation = givenDataFederation
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj
        self.__d, self.__N = self.__dataFederation.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__Xtn = np.zeros((self.__d, self.__N)) # d by N, local models
        self.__x = np.zeros(self.__d) # d by 1, global model
        pass
    
    def updateModelAtClient(self, givenEta):
        self.__aveGrads = self.__specificModelObj.computePersonalizedGradients()
        self.__Xtn = self.__Xtn - givenEta * self.__aveGrads
        return self.__Xtn
    
    def updateModelAtServer(self):
        self.__x = np.mean(self.__Xtn, axis=1)
        self.__Xtn = np.einsum('i,j->ij', self.__x, np.ones(self.__N)) 
        return self.__x
    
    def getPersonalizedModel(self):
        return self.__Xtn 
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfFedAvg))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfFedAvg))
        self.__name = 'FedAvg'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfFedAvg(givenDatasetObj=self.__dataFederation.getDatasetObj(), givenLambda=0.1)
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__specificModelObj = self.__initPersonalizedModel(givenName='federated logistic regression') # for example: convex clustering
        pass

    def __initPersonalizedModel(self, givenName):
        if givenName == 'federated logistic regression':
            specificModel = FederatedLogisticRegression(givenPersonalizedModels = np.zeros((self.__d, self.__N)), givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), 
                                                        givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), 
                                                        givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
        return specificModel    
    
    def executeBySGD(self):
        opt = Optimizer(givenDataFederation=self.__dataFederation, givenFederatedLearningSettings = self.__federatedLearningSettings, givenSpecificModelObj = self.__specificModelObj)
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
        federatedLearningSettings = FederatedLearningSettingsOfFedAvg()
        dataFederation = DataFederationOfFedAvg(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
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
        #datasetObj = IITnbswmbbfxycDataset(givenFn = '/home/yawei/data/medicalDatasets/real/classification/2型糖尿病视网膜病变风险预测.csv')
        Evaluater(givenDatasetObj=datasetObj).execute()
    pass





if __name__ == '__main__':
    np.random.seed(0)
    Test().testPersonalizedModel()






