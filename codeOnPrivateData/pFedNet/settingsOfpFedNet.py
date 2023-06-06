import csv
import numpy as np
from sklearn.neighbors import kneighbors_graph

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

class AuxiliaryMatrix(object):
    """
    The auxiliary matrix: :math:`\mathbf{M}` or :math:`\mathbf{N}`. For example, :math:`\mathbf{M}` looks like::

        1 0 0 
        0 1 0 
        0 0 0 
        0 0 1
        0 0 0 
        0 0 0 
    :math:`\mathbf{N}` looks like::

        0 0 0
        0 0 0
        1 0 0
        0 0 0
        0 1 0
        0 0 1
    """
    def __init__(self, givenFeatureIdList, givenNumOfFeatures):
        """
        givenFeatureIdList looks like: [1, 5, 8]
        """
        self.__featureIdList = givenFeatureIdList
        self.__numOfRows = givenNumOfFeatures
        self.__numOfCols = len(self.__featureIdList)
        self.__auxiliaryMat = self.__generate()

    def __generate(self):
        I = np.eye(self.__numOfRows)
        mat = np.zeros((self.__numOfRows, self.__numOfCols))
        for j in range(self.__numOfCols):
            mat[:,j] = I[:,self.__featureIdList[j]]
            pass
        return mat
    
    def get(self):
        return self.__auxiliaryMat
    pass

class MMatrix(object):
    """
    The auxiliary matrix: :math:`\mathbf{M}`. For example, :math:`\mathbf{M}` looks like::

        1 0 0
        0 1 0
        0 0 0
        0 0 1
        0 0 0
        0 0 0
    """
    def __init__(self, givenFeatureIdListOfSharingComponent, givenNumOfFeatures):
        """
        *givenFeatureIdListOfSharingComponent* looks like: [1, 5, 8]
        """
        self.__M = AuxiliaryMatrix(givenFeatureIdList=givenFeatureIdListOfSharingComponent, givenNumOfFeatures=givenNumOfFeatures).get()

    def get(self):
        return self.__M
    pass

class NMatrix(object):
    """
    The auxiliary matrix::math:`\mathbf{N}`. For example, :math:`\mathbf{N}` looks like::

        0 0 0
        0 0 0
        1 0 0
        0 0 0
        0 1 0
        0 0 1
    """
    def __init__(self, givenFeatureIdListOfPersonalizedComponent, givenNumOfFeatures):
        """
        *givenFeatureIdListOfPersonalizedComponent* looks like: [1, 5, 8]
        """
        self.__N = AuxiliaryMatrix(givenFeatureIdList=givenFeatureIdListOfPersonalizedComponent, givenNumOfFeatures=givenNumOfFeatures).get()

    def get(self):
        return self.__N
    pass

class QQMatrix(object):
    """
    Symetric and positive defined matrix. For example the auxiliary matrix may be :math:`\mathbf{Q}^\\top\mathbf{Q}` or :math:`\mathbf{Q}\mathbf{Q}^\\top`.
    """
    def __init__(self, givenQMatrix, givenTransposeOperatorLoc):
        if givenTransposeOperatorLoc == 'left':
            self.__mat = givenQMatrix.T @ givenQMatrix
        if givenTransposeOperatorLoc == 'right':
            self.__mat = givenQMatrix @ givenQMatrix.T
        eigValue, eigVec = np.linalg.eigh(self.__mat) # mat = P * Sigma * P^{-1}
        self.__sigma = eigValue
        self.__P = eigVec
        pass
    
    def get(self):
        return self.__mat
    
    def getEigenValue(self):
        return self.__sigma
    
    def getEigenVec(self):
        return self.__P
    pass

class QMatrix(object):
    """
    The difference matrix: :math:`\mathbf{Q}` (N by M), which is generated by graph :math:`\mathcal{G}`. 
    """
    def __init__(self, givenGraph):
        self.__graph = givenGraph
        self.__diffMat = self.__initDiffMat()
    
    def __initDiffMat(self):
        """
        Q: N nodes by M edges.
        """
        adjacentMat = self.__graph.getAdjacentMat()
        (N, N) = adjacentMat.shape
        edgeList = []
        for i in range(N): # for a client
            for j in range(i+1, N): # its neighbours
                if adjacentMat[i,j] == 1:
                    vec = np.zeros(N)
                    vec[i], vec[j] = 1, -1
                    edgeList.append(vec)
        QMat = np.transpose(np.array(edgeList))            
        return QMat

    def get(self):
        return self.__diffMat
    pass

class FormulationSettings(object):
    def __init__(self, givenDatasetObj, givenPersonalizedModelSettings, givenLambda):
        assert(isinstance(givenPersonalizedModelSettings, PersonalizedModelSettings))
        featureIdListOfSharingComponent = givenPersonalizedModelSettings.getFeatureIdListOfSharingComponent()
        featureIdListOfPersonalizedComponent = givenPersonalizedModelSettings.getFeatureIdListOfPersonalizedComponent()
        self.__d1, self.__d2 = len(featureIdListOfSharingComponent), len(featureIdListOfPersonalizedComponent)
        self.__d, self.__N = givenDatasetObj.getNumberOfFeatures(), givenDatasetObj.getNumberOfSamples()
        self.__MMat = MMatrix(givenFeatureIdListOfSharingComponent=featureIdListOfSharingComponent, givenNumOfFeatures=self.__d).get()
        self.__NMat = NMatrix(givenFeatureIdListOfPersonalizedComponent=featureIdListOfPersonalizedComponent, givenNumOfFeatures=self.__d).get()
        self.__lambda = givenLambda

    def getNumberOfSamples(self):
        return self.__N
    
    def getNumberOfFeatures(self):
        return self.__d

    def getNumberOfSharingFeatures(self):
        return self.__d1
    
    def getNumberOfPersonalizedFeatures(self):
        return self.__d2
    
    def getMMat(self):
        return self.__MMat
    
    def getNMat(self):
        return self.__NMat
    
    def getLambda(self):
        return self.__lambda
    pass

class PersonalizedModelSettings(object):
    def __init__(self, givenFeatureIdListOfSharingComponent, givenFeatureIdListOfPersonalizedComponent):
        self.__featureIdListOfSharingComponent = givenFeatureIdListOfSharingComponent
        self.__featureIdListOfPersonalizedComponent = givenFeatureIdListOfPersonalizedComponent
        pass
    
    def getFeatureIdListOfSharingComponent(self):
        return self.__featureIdListOfSharingComponent
    
    def getFeatureIdListOfPersonalizedComponent(self):
        return self.__featureIdListOfPersonalizedComponent
    
    def getNumberOfPersonalizedFeatures(self):
        return len(self.__featureIdListOfPersonalizedComponent)
    pass

class SketchMatrixForSimilarity(object):
    def __init__(self, givenSampleIdListOfClient, givenDatasetObj):
        """
        *sketchMat*: N by ?, every row represents a sketch of a node.
        """
        self.__type = 'label'
        self.__sampleIdListOfClient = givenSampleIdListOfClient
        self.__datasetObj = givenDatasetObj
        self.__numberOfClients = len(self.__sampleIdListOfClient)
        self.__sketchMat = self.__init() # N by ?
        pass
    
    def __init(self):
        if self.__type == 'data':
            dataSketch = self.__initDataSketch()
            return dataSketch
        if self.__type == 'label':
            labelSketch = self.__initLabelSketch()
            return labelSketch
        pass

    def __initDataSketch(self):
        N = len(self.__sampleIdListOfClient)
        dataMatListOfClient = []
        for clienti, sampleIdsOfClient in enumerate(self.__sampleIdListOfClient):
            dataMatListOfClient.append(self.__datasetObj.querySampleList(givenSampleIds = sampleIdsOfClient))
        sketchList = [np.mean(dataMatListOfClient[clienti], axis = 0) for clienti in range(N)]
        return sketchList
    
    def __initLabelSketch(self):
        allLabels = self.__datasetObj.getLabelVec()
        numberOfAllLabels = len(allLabels)
        labels = np.unique(allLabels) 
        labelRatioOfClients = np.zeros((self.__numberOfClients, len(labels)))
        for labeli, label in enumerate(labels):
            for clienti in range(self.__numberOfClients):
                labelsOfClient = [allLabels[sampleId] for sampleId in self.__sampleIdListOfClient[clienti]]
                labelRatioOfClients[clienti, labeli] = np.true_divide(np.sum(labelsOfClient == label), numberOfAllLabels) 
        return list(labelRatioOfClients)

    def get(self):
        return self.__sketchMat
    pass

class GraphGenerater(object):
    def __init__(self, givenDataFederation, givenGraphName):
        self.__name = givenGraphName
        self.__dataFederation = givenDataFederation
        pass
    
    def get(self):
        if self.__name == 'complete graph':
            return CompleteGraph(givenNumOfNode=self.__dataFederation.getNumberOfClients())
        if self.__name == 'start graph':
            return StarGraph(givenNumOfNode=self.__dataFederation.getNumberOfClients())
        if self.__name == 'similarity graph':
            return SimilarityGraph(givenSampleIdListOfClient= self.__dataFederation.getSampleIdListOfClient(), givenDatasetObj=self.__dataFederation.getDatasetObj())
    pass

class SimilarityGraph(object):
    """
    All nodes represent clients.
    """
    def __init__(self, givenSampleIdListOfClient, givenDatasetObj):
        self.__sampleIdListOfClient = givenSampleIdListOfClient
        self.__datasetObj = givenDatasetObj
        self.__numOfNeighbors = 3
        self.__adjacentMat = self.generateAdjacentMat()
        pass

    def generateAdjacentMat(self):
        sketchMatForSimilarity = SketchMatrixForSimilarity(givenSampleIdListOfClient=self.__sampleIdListOfClient, givenDatasetObj=self.__datasetObj)
        adjacentMat = kneighbors_graph(sketchMatForSimilarity.get(), self.__numOfNeighbors, mode = 'connectivity', include_self=False)
        return adjacentMat

    def getAdjacentMat(self):
        return self.__adjacentMat
    pass

class CompleteGraph(object):
    """
    All nodes represent clients.
    """
    def __init__(self, givenNumOfNode):
        self.__N = givenNumOfNode
        self.__adjacentMat = self.generateAdjacentMat()
        pass

    def generateAdjacentMat(self):
        adjacentMat = np.ones((self.__N, self.__N)) - np.eye(self.__N)
        return adjacentMat

    def getAdjacentMat(self):
        return self.__adjacentMat
    pass

class StarGraph(object):
    """
    All nodes represent clients.
    """
    def __init__(self, givenNumOfNode):
        self.__N = givenNumOfNode
        self.__adjacentMat = self.generateAdjacentMat()
        pass

    def generateAdjacentMat(self):
        adjacentMat = np.zeros((self.__N, self.__N))
        adjacentMat[:,0] = np.ones(self.__N)
        adjacentMat[0,:] = np.ones(self.__N)
        adjacentMat[0,0] = 0 
        return adjacentMat

    def getAdjacentMat(self):
        return self.__adjacentMat
    pass




















