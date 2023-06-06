import numpy as np 
from settingsOfFPFC import FormulationSettings as FormulationSettingsOfFPFC
import os, sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from settings import DataFederation as DataFederationOfFPFC, FederatedLearningSettings as FederatedLearningSettingsOfFPFC
from tools import ModelLogger

class Optimizer(object):
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfFPFC))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfFPFC))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfFPFC))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__T = 100
        self.__Omega, self.__Zeta = np.zeros((self.__d, self.__N)), np.zeros((self.__d, self.__N)) 
        self.__Delta, self.__Theta, self.__V = np.zeros((self.__N, self.__N, self.__d)), np.zeros((self.__N, self.__N, self.__d)), np.zeros((self.__N, self.__N, self.__d))
        self.__rho = 1.0
        pass
    
    def getPersonalizedModel(self):
        return self.__Omega
    
    def __updatePersonalizedModel(self):
        self.__specificModelObj.updatePersonalziedModels(givenXtn = self.__Omega)
        pass

    def __updateOmega(self):
        """
        Omega is updated at clients.
        """
        alpha = 1e-3
        presonalizedGrad = self.__specificModelObj.computePersonalizedGradients()
        for t in range(0, self.__T):
            self.__Omega = self.__Omega - alpha * (presonalizedGrad + self.__rho * (self.__Omega - self.__Zeta))
        return self.__Omega

    def __updateDelta(self):
        """
        Delta is updated at server.
        """
        for clienti in range(0, self.__N):
            for clientj in range(clienti+1, self.__N):
                valij = self.__Omega[:,clienti] - self.__Omega[:,clientj] + self.__V[clienti, clientj] / self.__rho
                self.__Delta[clienti, clientj] = valij
        return self.__Delta
    
    def __updateTheta(self):
        """
        Theta is updated at server.
        """
        xi, lambdav, a = 1.0, 1e-1, 1e-1
        for clienti in range(0, self.__N):
            for clientj in range(clienti+1, self.__N):
                deltaNorm = np.linalg.norm(self.__Delta[clienti, clientj])
                if deltaNorm <= xi + lambdav / self.__rho:
                    self.__Theta[clienti, clientj] = xi * self.__rho / (lambdav+xi*self.__rho) * self.__Delta[clienti, clientj]
                if deltaNorm >= xi + lambdav / self.__rho and deltaNorm <= lambdav + lambdav / self.__rho:
                    self.__Theta[clienti, clientj] = (1 - lambdav / (self.__rho * deltaNorm)) * self.__Delta[clienti, clientj]
                if deltaNorm > lambdav + lambdav / self.__rho and deltaNorm <= a * lambdav:
                    self.__Theta[clienti, clientj] = (np.maximum(0, 1 - a*lambdav / (a-1)*self.__rho*deltaNorm)) / (1-1.0/((a-1)*self.__rho)) * self.__Delta[clienti, clientj]
                if deltaNorm > a * lambdav:
                    self.__Theta[clienti, clientj] = self.__Delta[clienti, clientj]
                self.__Theta[clientj, clienti] = -1.0 * self.__Theta[clienti, clientj]
        return self.__Theta
    
    def __updateV(self):
        """
        V is updated at server.
        """
        for clienti in range(0, self.__N):
            for clientj in range(clienti+1, self.__N):
                self.__V[clienti, clientj] = self.__V[clienti, clientj] + self.__rho * (self.__Omega[:,clienti] - self.__Omega[:,clientj] - self.__Theta[clienti, clientj])
                self.__V[clientj, clienti] = -1.0 * self.__V[clienti, clientj]
        return self.__V
    
    def __updateZeta(self):
        """
        Zeta is updated at server.
        """
        for clienti in range(0, self.__N):
            temp = np.zeros(self.__d)
            for clientj in range(0, self.__N):
                temp += self.__Omega[:, clientj] + self.__Theta[clienti, clientj] - self.__V[clienti, clientj] / self.__rho
            self.__Zeta[:,clienti] = temp / self.__N
        return self.__Zeta
    
    def updateModelAtClient(self):
        self.__updateOmega()
        pass
    
    def updateModelAtServer(self):
        self.__updateDelta()
        self.__updateTheta()
        self.__updateV()
        self.__updateZeta()
        self.__updatePersonalizedModel()
        pass
    pass

class PersonalizedFederatedLearning(object):
    """
    Ref.: Clustered Federated Learning based on Nonconvex Pairwise Fusion. 2022.11
    """
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfFPFC))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfFPFC))
        self.__name = 'FPFC'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfFPFC(givenDatasetObj=self.__dataFederation.getDatasetObj())
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
        federatedLearningSettings = FederatedLearningSettingsOfFPFC()
        dataFederation = DataFederationOfFPFC(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
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


