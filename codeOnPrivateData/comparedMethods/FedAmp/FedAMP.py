import numpy as np 
import cvxpy as cp
from settingsOfFedAMP import FormulationSettings as FormulationSettingsOfFedAMP
import os,sys
sys.path.append("/home/yawei/communication-efficient-federated-training/code")
from models import ConvexClustering, ClusterPath, FederatedLogisticRegression
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm
from settings import DataFederation as DataFederationOfFedAMP, FederatedLearningSettings as FederatedLearningSettingsOfFedAMP
from tools import ModelLogger

class CvxOptimizer(object):
    """
    The cvx optimizer to solve local models (for clients). No global models (for servers). Ref: Personalized Cross-Silo Federated Learning on Non-IID Data.
    
    Alternative optimizing:
    min_{W} G(W) := F(W) + lambda sum_{i<j} A(||Wi - Wj||^2)
    """
    def __init__(self, givenDataFederation, givenFormulationSettings, givenFederatedLearningSettings, givenSpecificModelObj):
        assert(isinstance(givenDataFederation, DataFederationOfFedAMP))
        assert(isinstance(givenFormulationSettings, FormulationSettingsOfFedAMP))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfFedAMP))
        self.__dataFederation = givenDataFederation
        self.__formulationSettings = givenFormulationSettings
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__specificModelObj = givenSpecificModelObj # for example: convex clustering
        self.__lambda, self.__alpha = self.__formulationSettings.getLambda(), 1e-3
        self.__d, self.__N = self.__formulationSettings.getNumberOfFeatures(), self.__federatedLearningSettings.getNumOfClients()
        self.__W = np.zeros((self.__d, self.__N)) # d by N, personalized model
        self.__U = np.zeros((self.__d, self.__N)) # d by N
        pass
    
    def getPersonalizedModel(self):
        return self.__W
    
    def __computeDerivativeOfAw(self, givenInputScalar):
        """
        A(u) := 1 - exp(-givenInputScalar/sigma)
        """
        sigma = 1.0
        result = -np.exp(-givenInputScalar / sigma) * (-1.0 / sigma)
        return result
    
    def __computeXi1(self, givenAlpha, givenW):
        xiForWi = []
        for clienti in range(self.__N):
            diffList = [(np.linalg.norm(self.__W[:,clienti] - self.__W[:,columni]) ** 2) for columni in range(self.__N)]
            diffList.pop(clienti) # remove the i-th element
            gradList = [self.__computeDerivativeOfAw(ele) for ele in diffList]
            sumOfGrads = np.sum(gradList)
            xiForWi.append((1 - givenAlpha * sumOfGrads) * givenW[:, clienti])
        return xiForWi # N by d
    
    def __computeXij(self, givenAlpha, givenW):
        xiForWj = []
        for clienti in range(self.__N):
            diffList = [(np.linalg.norm(self.__W[:,clienti] - self.__W[:,columni]) ** 2) for columni in range(self.__N)]
            gradOfAwList = [self.__computeDerivativeOfAw(ele) for ele in diffList]
            gradList = [gradOfAwList[columni] * givenW[:, columni] for columni in range(self.__N)]
            gradList.pop(clienti) # remove the i-th element
            sumOfGrads = np.sum(gradList, axis = 0)
            xiForWj.append(givenAlpha * sumOfGrads)
        return xiForWj # d
    
    def __updateByOptimizeAw(self):
        """
        Perform U_k = W_{k-1} - alpha_k nabla A(W_{k-1}) according to Eq. 5.
        """
        left = self.__computeXi1(givenAlpha=self.__alpha, givenW=self.__W) 
        right = self.__computeXij(givenAlpha=self.__alpha, givenW=self.__W)
        self.__U = np.array([left[clienti] + right[clienti] for clienti in range(self.__N)]).T
        return self.__U
    
    def updateModelAtClient(self):
        self.__updateByOptimizeFw()
        pass
    
    def updateModelAtServer(self):
        self.__updateByOptimizeAw()
        pass
    
    def __updateByOptimizeFw(self):
        W = np.zeros((self.__d, self.__N))
        for i in range(30):
            eta = 1e-2
            flr = FederatedLogisticRegression(givenPersonalizedModels = W, givenSampleMat = self.__dataFederation.getDatasetObj().getSampleMat(), givenLabelVec = self.__dataFederation.getDatasetObj().getLabelVec(), 
                                          givenTrainingSampleIdListOfClient = self.__dataFederation.getSampleIdListOfClient(), givenTestSampleIdListOfClient = self.__dataFederation.getTestSampleIdListOfClient())
            grad = flr.computePersonalizedGradients() + self.__lambda * (W - self.__U)
            W = W - eta/np.sqrt(i+1) * grad
            pass
        self.__W = W
        return self.__W 
    pass

class PersonalizedFederatedLearning(object):
    def __init__(self, givenDataFederation, givenFederatedLearningSettings):
        assert(isinstance(givenDataFederation, DataFederationOfFedAMP))
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettingsOfFedAMP))
        self.__name = 'FedAMP'
        self.__dataFederation = givenDataFederation 
        self.__formulationSettings = FormulationSettingsOfFedAMP(givenDatasetObj=self.__dataFederation.getDatasetObj(), givenLambda = 0.1) 
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
    
    def executeByCvx(self):
        co = CvxOptimizer(givenDataFederation=self.__dataFederation, givenFormulationSettings=self.__formulationSettings, givenFederatedLearningSettings=self.__federatedLearningSettings, givenSpecificModelObj=self.__specificModelObj)
        for i in range(self.__federatedLearningSettings.getNumOfIterations()):
            self.__specificModelObj.appendRecords()
            # update x
            co.updateModelAtClient()
            # update Z
            co.updateModelAtServer()
            self.__specificModelObj.updatePersonalziedModels(givenXtn = co.getPersonalizedModel())
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
        federatedLearningSettings = FederatedLearningSettingsOfFedAMP()
        dataFederation = DataFederationOfFedAMP(givenDatasetObj=self.__datasetObj, givenFederatedLearningSettings=federatedLearningSettings)
        pcc = PersonalizedFederatedLearning(givenDataFederation=dataFederation,  givenFederatedLearningSettings=federatedLearningSettings)
        pcc.executeByCvx()
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




if __name__ == '__main__':
    np.random.seed(0)
    Test().testPersonalizedModel()







































