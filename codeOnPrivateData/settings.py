import os
import csv
import numpy as np
from sklearn.cluster import KMeans

class CsvFile(object):
    def __init__(self, givenFilePath, hasHeader = False):
        self.__filePath = givenFilePath
        self.__recordList = self.read(hasHeader=hasHeader)
    
    def read(self, hasHeader = False):
        resultList = []
        with open(self.__filePath) as f:
            reader = csv.reader(f) 
            if hasHeader == True:
                headerList = [column.strip() for column in str(next(reader)).split(' ')]
            for row in reader: # read file line by line.
                resultList.append(row)
        return resultList

    def getRecords(self):
        return self.__recordList
    pass

class DataFederation(object):
    def __init__(self, givenDatasetObj, givenFederatedLearningSettings):
        assert(isinstance(givenFederatedLearningSettings, FederatedLearningSettings))
        self.__datasetObj = givenDatasetObj # numberOfSamples by numberOfFeatures
        self.__federatedLearningSettings = givenFederatedLearningSettings
        self.__sampleIdListOfClient = self.__generateSampleIdListForClients()
        self.__testSampleIdListOfClient = self.__initTestSampleIdListOfClient()
        pass
    
    def __initTestSampleIdListOfClient(self):
        testSampleIdListOfClient = []
        for clienti, sampleIds in enumerate(self.__sampleIdListOfClient): 
            numOfTestSampleIds = int(len(sampleIds) * 0.2)
            testSampleIds = []
            for i in range(numOfTestSampleIds):
                sampleId = self.__sampleIdListOfClient[clienti].pop()
                testSampleIds.append(sampleId)
            testSampleIdListOfClient.append(testSampleIds)
        return testSampleIdListOfClient

    def __generateSampleIdListForClientsByEqualPartation(self):
        sampleIdsOfClientList = []
        N = self.__federatedLearningSettings.getNumOfClients()
        for clienti in range(N):
            moduleOfSamples = int(np.true_divide(self.getNumberOfSamples(), N))
            startidForClient, endidForClient = clienti * moduleOfSamples, np.min([self.getNumberOfSamples(), (clienti+1) * moduleOfSamples])
            sampleIdsOfClientList.append(list(range(startidForClient, endidForClient)))
        return sampleIdsOfClientList
    
    def __generateSampleIdListForClientsByClustering(self):
        N = self.__federatedLearningSettings.getNumOfClients()
        dataMat = self.__datasetObj.getSampleMat()
        kmeans = KMeans(n_clusters=N).fit(dataMat)
        label = kmeans.labels_
        sampleIdsOfClientList = []
        for clienti in range(N):
            sampleIds = np.where(label == clienti)
            sampleIdsOfClientList.append(sampleIds[0].tolist())
        return sampleIdsOfClientList

    def __generateSampleIdListForClientsByAffine(self):
        """
        Every client has only **ONE** sample. Defaultly, numberOfClients = numberOfSamples.
        """
        N = self.__federatedLearningSettings.getNumOfClients()
        sampleIdsOfClientList = [[clienti] for clienti in range(N)]
        return sampleIdsOfClientList
    
    def __generateSampleIdListForClientsByLabelRatio(self, givenLabelRatioOfClients):
        """
        Given label ratio of every client. 
        *givenLabelRatioOfClients*: numberOfClients by numberOfLabels
        """
        numberOfClients = self.__federatedLearningSettings.getNumOfClients()
        numberOfLabels = len(self.__datasetObj.getLabelVec())
        allLabels = self.__datasetObj.getLabelVec()
        labels = np.unique(allLabels)
        assert(givenLabelRatioOfClients.shape == (numberOfClients, len(labels)))
        resultList, sampleIdsOfLabel = [], allLabels
        for labeli, label in enumerate(labels): 
            sampleIdsOfClient = []
            sampleIdsOfLabel = list(np.where(allLabels == label)[0]) # select all sample ids for every label.
            numOfLabeli = np.sum(allLabels == label)
            for clienti in range(numberOfClients): 
                numOfSampleIds = min(int(givenLabelRatioOfClients[clienti, labeli]*numOfLabeli), len(sampleIdsOfLabel))
                sampleIds = [sampleIdsOfLabel.pop() for i in range(numOfSampleIds)] 
                sampleIdsOfClient.append(sampleIds)
            if labeli == 0:
                resultList.extend(sampleIdsOfClient)
            else:
               resultList = [resultList[i] + sampleIdsOfClient[i] for i in range(numberOfClients)]
        return resultList # numberOfClients by ?

    def __generateSampleIdListForClients(self, givenType = 'clustering'):
        if givenType == 'equal partation':
            sampleIds = self.__generateSampleIdListForClientsByEqualPartation()
        if givenType == 'affine':
            sampleIds = self.__generateSampleIdListForClientsByAffine()
        if givenType == 'clustering':
            sampleIds = self.__generateSampleIdListForClientsByClustering()
        if givenType == 'label ratio':
            labelRatios = self.__loadConfigSettingsOfLabelRatio()
            sampleIds = self.__generateSampleIdListForClientsByLabelRatio(givenLabelRatioOfClients=labelRatios)
        return sampleIds
    
    def __loadConfigSettingsOfLabelRatio(self):
        """
        *labelUnbalanceConfig.csv*
        
        *labelUnbalanceConfig2.csv*
        """
        configPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config', 'labelUnbalanceConfig3.csv')
        labelRatioRecords = CsvFile(givenFilePath=configPath, hasHeader=True).getRecords()
        resultList = []
        for labelsOfRow in labelRatioRecords:
            row = [float(ele) for ele in labelsOfRow[1:]] # the first element is class name, remove it
            resultList.append(row)
        return np.array(resultList)
    
    def getSampleIdListOfClient(self):
        return self.__sampleIdListOfClient
    
    def getTestSampleIdListOfClient(self):
        return self.__testSampleIdListOfClient
    
    def getNormalizedWeightsForClients(self):
        sampleIdList = self.getSampleIdListOfClient()
        numOfSamplesList = [len(idList) for idList in sampleIdList]
        allEffectiveSamples = np.sum(numOfSamplesList)
        normalizedWeights = [np.true_divide(ele, allEffectiveSamples) for ele in numOfSamplesList]
        return normalizedWeights
    
    def getDatasetObj(self):
        return self.__datasetObj
    
    def querySample(self, givenSampleId):
        return self.__datasetObj.querySample(givenSampleId)
    
    def queryLabel(self, givenSampleId):
        return self.__datasetObj.queryLabel(givenSampleId)
    
    def querySampleIds(self, givenClientId):
        sampleIds = self.__sampleIdListOfClient[givenClientId]
        return sampleIds
    
    def getNumberOfSamples(self):
        return self.__datasetObj.getNumberOfSamples()
    
    def getNumberOfFeatures(self):
        return self.__datasetObj.getNumberOfFeatures()
    pass

class FederatedLearningSettings(object):
    def __init__(self):
        self.__numOfClients = 20
        self.__numOfServers = 1
        self.__numOfIterations = 10

    def getNumOfClients(self):
        return self.__numOfClients
    
    def getNumOfServers(self):
        return self.__numOfServers
    
    def getNumOfIterations(self):
        return self.__numOfIterations
    pass


