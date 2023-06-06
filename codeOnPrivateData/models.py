import numpy as np
import cvxpy as cp
from sklearn.metrics import balanced_accuracy_score, accuracy_score

class ConvexClustering(object):
    def __init__(self):
        pass

    def computeStochasticGradient(self, givenModelVec, givenSample):
        """
        Stochastic gradient of :math::`f_n(\mathbf{x};\mathbf{a}) := \\left\| \mathbf{x} - \mathbf{a} \\right\|^2`::

        \mathbf{g} := 2 (\mathbf{x} - \mathbf{a})
        """
        stochasticGrad = 2*(givenModelVec - givenSample)
        return stochasticGrad   
    
    def defineFunction(self, givenXtn, givenSampleMat, givenLabelVec, givenSampleIdListOfClient):
        """
        *givenXth*: d by N
        *givenSketchMat*: N by d

        Define the cumulative loss::

        \sum_{n=1}^N f_n(\X^{(n)}, \A)
        """
        N, numOfSamples, fun = givenXtn.shape[1], givenSampleMat.shape[0], 0
        weightOfLocalLoss = np.true_divide(N, numOfSamples)
        for n in range(N):
            funOfClient = 0
            for sampleId in givenSampleIdListOfClient[n]:
                funOfClient += cp.norm2(givenXtn[:,n] - givenSampleMat[sampleId,:].T) ** 2
            fun = fun + weightOfLocalLoss * funOfClient
        return fun
    
    pass

class StandardLogisticRegression(object):
    def __init__(self, givenModel, givenSampleMat, givenLabelVec):
        self.__x = givenModel # d
        self.__sampleMat = givenSampleMat # numOfSamples by d
        self.__labelVec = givenLabelVec
        self.__numOfSamples, self.__d = givenSampleMat.shape[0], givenSampleMat.shape[1]
        self.__regCoeff = 1 # logistic regression with l1-norm, which is used for feature selection.
        pass
    
    def __computeStochasticGradientByData(self, givenSample, givenLabel):
        stochasticGrad = - givenLabel * givenSample / (1 + np.exp(givenLabel*np.dot(givenSample, self.__x))) + self.__regCoeff * np.sign(self.__x)
        return stochasticGrad
    
    def __computeStochasticGradientById(self, givenSampleId):
        sample, label = self.__sampleMat[givenSampleId,:], self.__labelVec[givenSampleId]
        stochasticGrad = self.__computeStochasticGradientByData(givenSample=sample, givenLabel=label)
        return stochasticGrad       
    
    def computeStochasticGradient(self):
        sampleId = np.random.randint(0,self.__numOfSamples)
        stochasticGrad = self.__computeStochasticGradientById(givenSampleId=sampleId)
        return stochasticGrad
    
    def computeAverageGradient(self):
        temp0 = np.einsum('ij,j->i', self.__sampleMat, self.__x) 
        temp1 = - self.__labelVec / (1 + np.exp(np.einsum('i,i->i', self.__labelVec, temp0))) # numOfSamples by 1
        grads = np.einsum('i,ij->ij', temp1, self.__sampleMat) # numOfSamples by d
        aveGrad = np.mean(grads, axis=0) + self.__regCoeff * np.sign(self.__x)# d
        return aveGrad
    
    def defineLossFunction(self):
        fun = 0
        for i in range(self.__numOfSamples):
            temp = - self.__labelVec[i] * cp.scalar_product(self.__x, self.__sampleMat[i,:])
            fun += cp.logistic(temp) + self.__regCoeff * cp.norm1(self.__x)
        return fun / self.__numOfSamples
    
    def getLoss(self):
        fun = self.defineLossFunction()
        loss = fun.value
        return loss
    
    def getAccuracy(self):
        predAccVal = 0
        for idOfSample in range(self.__numOfSamples):
            temp = np.dot(self.__x, self.__sampleMat[idOfSample,:])
            probOfPredLabel = np.true_divide(1, 1 + np.exp(-temp))
            if probOfPredLabel >= 0.5:
                predLabel = 1
            if probOfPredLabel < 0.5:
                predLabel = -1
            if predLabel == self.__labelVec[idOfSample]:
                predAccVal += 1
        return np.true_divide(predAccVal, self.__numOfSamples)
    pass

class FederatedLogisticRegression(object):
    def __init__(self, givenPersonalizedModels, givenSampleMat, givenLabelVec, givenTrainingSampleIdListOfClient, givenTestSampleIdListOfClient):
        self.__personalizedModels = givenPersonalizedModels
        self.__sampleMat = givenSampleMat # N by d
        self.__labelVec = givenLabelVec
        self.__trainingSampleIdListOfClient = givenTrainingSampleIdListOfClient
        self.__testSampleIdListOfClient = givenTestSampleIdListOfClient
        self.__d, self.__N = self.__personalizedModels.shape[0], self.__personalizedModels.shape[1]
        self.__testClassicAccuracyRecords, self.__testBalancedAccuracyRecords = [], []
        self.__trainingClassicAccuracyRecords, self.__trainingBalancedAccuracyRecords = [], []
        self.__testLossRecords, self.__trainingLossRecords = [], [] 
        pass
    
    def updatePersonalziedModels(self, givenXtn):
        self.__personalizedModels = givenXtn
        return
    
    def getNormalizedWeight(self, givenType):
        if givenType == 'training':
            totalNumberOfIds = sum([len(idList) for idList in self.__trainingSampleIdListOfClient])
            normalizedWeights = [np.true_divide(len(idList), totalNumberOfIds) for idList in self.__trainingSampleIdListOfClient]
        if givenType == 'test':
            totalNumberOfIds = sum([len(idList) for idList in self.__testSampleIdListOfClient])
            normalizedWeights = [np.true_divide(len(idList), totalNumberOfIds) for idList in self.__testSampleIdListOfClient]
        return normalizedWeights
    
    def computePersonalizedGradients(self):
        stocGradMat = self.fetchPersonalziedGradient(givenPersonalizedModel=self.__personalizedModels)
        return stocGradMat
    
    def fetchPersonalziedGradient(self, givenPersonalizedModel):
        """
        *givenPersonalizedModel*: d by N
        """
        stocGradList = []
        normalizedWeights = self.getNormalizedWeight(givenType='training')
        for clienti, sampleIdListOfClient in enumerate(self.__trainingSampleIdListOfClient):
            slr = StandardLogisticRegression(givenModel=givenPersonalizedModel[:,clienti], givenSampleMat=self.__sampleMat[sampleIdListOfClient,:], givenLabelVec=self.__labelVec[sampleIdListOfClient])
            stocGrad = slr.computeAverageGradient() # average for local samples
            stocGradList.append(normalizedWeights[clienti]*stocGrad) 
        stocGradMat = np.array(stocGradList)
        return stocGradMat.T # d by N
    
    def defineLocalLossFunction(self, givenClientId, givenType = 'training'):
        if givenType == 'training':
            localSampleMat = self.__sampleMat[self.__trainingSampleIdListOfClient[givenClientId]]
            localLabels = self.__labelVec[self.__trainingSampleIdListOfClient[givenClientId]]
        if givenType == 'test':
            localSampleMat = self.__sampleMat[self.__testSampleIdListOfClient[givenClientId]]
            localLabels = self.__labelVec[self.__testSampleIdListOfClient[givenClientId]]
        slr = StandardLogisticRegression(givenModel=self.__personalizedModels[:,givenClientId], givenSampleMat=localSampleMat, givenLabelVec=localLabels)
        localLossFun = slr.defineLossFunction()
        return localLossFun
    
    def getLocalLoss(self, givenClientId, givenType = 'training'):
        funOfClient = self.defineLocalLossFunction(givenClientId, givenType)
        lossOfClient = funOfClient.value
        return lossOfClient
    
    def getLocalLossFunctionList(self, givenType = 'training'):
        lossFunList = []
        normalizedWeights = self.getNormalizedWeight(givenType)
        for n in range(self.__N):
            funOfClient = self.defineLocalLossFunction(givenClientId=n, givenType = givenType)    
            lossFunList.append(funOfClient * normalizedWeights[n]) 
        return lossFunList # N 
    
    def defineObjectiveFunction(self):
        lossFunList = self.getLocalLossFunctionList(givenType = 'training')
        return cp.sum(lossFunList)
    
    def getObjectiveLoss(self):
        lossFun = self.defineObjectiveFunction()
        lossVal = lossFun.value
        return lossVal
    
    def getTrainingLoss(self):
        trainingLoss = self.getObjectiveLoss()
        return trainingLoss
    
    def getTestLoss(self):
        testLossFunList = self.getLocalLossFunctionList(givenType='test')
        testLossFun = cp.sum(testLossFunList)
        return testLossFun.value
    
    def __fetchPredictiveLabels(self, givenPersonalizedModel, givenSampleIdListOfClient):
        predLabels, trueLabels = [], []
        for clienti in range(self.__N):
            for id in givenSampleIdListOfClient[clienti]:
                temp = np.dot(givenPersonalizedModel[:, clienti], self.__sampleMat[id,:])
                probOfPredLabel = np.true_divide(1, 1 + np.exp(-temp))
                if probOfPredLabel >= 0.5:
                    predLabels.append(1)
                if probOfPredLabel < 0.5:
                    predLabels.append(-1)
                trueLabels.append(self.__labelVec[id])
        return predLabels, trueLabels
    
    def fetchClassicAccuracy(self, givenPersonalizedModel, givenSampleIdListOfClient):
        predLabels, trueLabels = self.__fetchPredictiveLabels(givenPersonalizedModel, givenSampleIdListOfClient)
        acc = accuracy_score(y_true=trueLabels, y_pred=predLabels)
        return acc

    def fetchBalancedAccuracy(self, givenPersonalizedModel, givenSampleIdListOfClient):
        predLabels, trueLabels = self.__fetchPredictiveLabels(givenPersonalizedModel, givenSampleIdListOfClient)
        bacc = balanced_accuracy_score(y_true=trueLabels, y_pred=predLabels)
        return bacc
    
    def getTrainingAccuracy(self):
        trainingAcc = self.fetchClassicAccuracy(givenPersonalizedModel=self.__personalizedModels, givenSampleIdListOfClient=self.__trainingSampleIdListOfClient)
        return trainingAcc
    
    def __appendTestAccuracyRecords(self):
        classicRecord = self.fetchClassicAccuracy(givenPersonalizedModel=self.__personalizedModels, givenSampleIdListOfClient=self.__testSampleIdListOfClient)
        balancedRecord = self.fetchBalancedAccuracy(givenPersonalizedModel=self.__personalizedModels, givenSampleIdListOfClient=self.__testSampleIdListOfClient)
        print("test classic accuracy: {%.4f} | test balanced accuracy: {%.4f}" % (classicRecord, balancedRecord))
        self.__testClassicAccuracyRecords.append(classicRecord)
        self.__testBalancedAccuracyRecords.append(balancedRecord)
        pass
    
    def __appendTrainingAccuracyRecords(self):
        classicRecord = self.fetchClassicAccuracy(givenPersonalizedModel=self.__personalizedModels, givenSampleIdListOfClient=self.__trainingSampleIdListOfClient)
        balancedRecord = self.fetchBalancedAccuracy(givenPersonalizedModel=self.__personalizedModels, givenSampleIdListOfClient=self.__trainingSampleIdListOfClient)
        print("training classic accuracy: {%.4f} | training balanced accuracy: {%.4f}" % (classicRecord, balancedRecord))
        self.__trainingClassicAccuracyRecords.append(classicRecord)
        self.__trainingBalancedAccuracyRecords.append(balancedRecord)
        pass
    
    def __appendTestLossRecords(self):
        record = self.getTestLoss()
        print("test loss: {%.4f}" % record)
        self.__testLossRecords.append(record)
        return record
    
    def __appendTrainingLossRecords(self):
        record = self.getTrainingLoss()
        print("training loss: {%.4f}" % record)
        self.__trainingLossRecords.append(record)
        return record
    
    def appendAccuracyRecords(self, givenType = 'all'):
        """
        *givenType*: 'test accuracy', 'training accuracy', 'all'
        """
        if givenType == 'test accuracy':
            self.__appendTestAccuracyRecords()
        if givenType == 'training accuracy':
            self.__appendTrainingAccuracyRecords()
        if givenType == 'all':
            self.__appendTestAccuracyRecords()
            self.__appendTrainingAccuracyRecords()
        pass
    
    def appendLossRecords(self, givenType = 'all'):
        """
        *givenType*: 'test loss', 'training loss', 'all'
        """
        if givenType == 'test loss':
            self.__appendTestLossRecords()
        if givenType == 'training loss':
            self.__appendTrainingLossRecords()
        if givenType == 'all':
            self.__appendTestLossRecords()
            self.__appendTrainingLossRecords()
        pass
    
    def appendRecords(self, givenType = 'all'):
        """
        *givenType*: 'loss', 'accuracy', 'all'
        """
        if givenType == 'loss':
            self.appendLossRecords()
        if givenType == 'accuracy':
            self.appendAccuracyRecords()
        if givenType == 'all':
            self.appendLossRecords()
            self.appendAccuracyRecords()
        pass
    
    def getTestClassicAccuracyRecords(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__testClassicAccuracyRecords
        if givenType == 'str':
            strList = [','.join([str(val) for val in self.__testClassicAccuracyRecords])]
            return "\n".join(strList)
        pass
    
    def getTestBalancedAccuracyRecords(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__testBalancedAccuracyRecords
        if givenType == 'str':
            strList = [','.join([str(val) for val in self.__testBalancedAccuracyRecords])]
            return "\n".join(strList)
        pass
    
    def getTrainingClassicAccuracyRecords(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__trainingClassicAccuracyRecords
        if givenType == 'str':
            strList = [','.join([str(val) for val in self.__trainingClassicAccuracyRecords])]
            return "\n".join(strList)
        pass

    def getTrainingBalancedAccuracyRecords(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__trainingBalancedAccuracyRecords
        if givenType == 'str':
            strList = [','.join([str(val) for val in self.__trainingBalancedAccuracyRecords])]
            return "\n".join(strList)
        pass

    def getTestLossRecords(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__testLossRecords
        if givenType == 'str':
            strList = [','.join([str(val) for val in self.__testLossRecords])]
            return "\n".join(strList)
        pass
    
    def getTrainingLossRecords(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__trainingLossRecords
        if givenType == 'str':
            strList = [','.join([str(val) for val in self.__trainingLossRecords])]
            return "\n".join(strList)
        pass
    
    def getPersonalizedModel(self, givenType = 'float'):
        """
        *givenType*: 'float', 'str'
        """
        if givenType == 'float':
            return self.__personalizedModels
        if givenType == 'str':
            modelStrList = [','.join([str(val) for val in self.__personalizedModels[i,:]]) for i in range(self.__d)]
            return "\n".join(modelStrList)
        pass
    pass






