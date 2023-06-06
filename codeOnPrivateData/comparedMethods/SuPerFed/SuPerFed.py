import numpy as np 
from settingsOfSuPerFed import FormulationSettings as FormulationSettingsOfSuPerFed
import os, sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from settings import DataFederation as DataFederationOfSuPerFed, FederatedLearningSettings as FederatedLearningSettingsOfSuPerFed
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from tools import ModelLogger

class Optimizer(object):
    """
    The optimizer to solve local models (for clients), NO global models (for servers).
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfSuPerFed))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfSuPerFed))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfSuPerFed))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__Wf, self.__Wl, self.__wg = np.zeros((self.__d, self.__N)), np.zeros((self.__d, self.__N)), np.zeros(self.__d)
        pass
    
    def getPersonalizedModel(self):
        return self.__Wf
    
    def __computeGradsWrtWf(self, givenW, givenLambda, givenMu, givenNu):
        Wg = np.einsum('i,j->ij', self.__wg, np.ones(self.__N))
        temp = np.diagonal(self.__Wf.T @ self.__Wl)
        gradPartOfCos1 = 2*np.einsum('i,i->i', np.cos(temp), np.sin(temp))
        gradPartOfCos = np.einsum('i,ji->ji', gradPartOfCos1, self.__Wf)
        flr = FederatedLogisticRegression(givenPersonalizedModels = givenW, givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), 
                                            givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), 
                                            givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
        gradWf = (1-givenLambda) * flr.computePersonalizedGradients() + givenMu * (2*(self.__Wf - Wg)) + givenNu*gradPartOfCos 
        return gradWf
    
    def __computeGradsWrtWl(self, givenW, givenLambda, givenNu):
        temp = np.diagonal(self.__Wf.T @ self.__Wl)
        gradPartOfCos1 = 2*np.einsum('i,i->i', np.cos(temp), np.sin(temp))
        gradPartOfCos = np.einsum('i,ji->ji', gradPartOfCos1, self.__Wl)
        flr = FederatedLogisticRegression(givenPersonalizedModels = givenW, givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), 
                                            givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), 
                                            givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
        gradWl = givenLambda * flr.computePersonalizedGradients() + givenNu * gradPartOfCos
        return gradWl
    
    def updateModelAtClient(self, givenEta):
        lambdav = np.random.random() # uniform sampling from [0, 1].
        mu, nu = 1e-1, 1e-1
        W = (1-lambdav) * self.__Wf + lambdav * self.__Wl
        GradsWrtWf = self.__computeGradsWrtWf(givenW = W, givenLambda = lambdav, givenMu = mu, givenNu = nu)
        GradsWrtWl = self.__computeGradsWrtWl(givenW = W, givenLambda = lambdav, givenNu = nu)
        self.__Wf = self.__Wf - givenEta * GradsWrtWf
        self.__Wl = self.__Wl - givenEta * GradsWrtWl
        return 
    
    def updateModelAtServer(self):
        normalizedWights = self.__dataFederation.getNormalizedWeightsForClients()
        self.__wg = np.einsum('j,ij->i', normalizedWights, self.__Wf)
        return self.__wg
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfSuPerFed))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfSuPerFed))
        self.__name = 'SuPerFed'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfSuPerFed(givenDatasetObj=self.__dataFederation.getDatasetObj()) 
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
        federatedLearningSettings = FederatedLearningSettingsOfSuPerFed()
        dataFederation = DataFederationOfSuPerFed(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
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







