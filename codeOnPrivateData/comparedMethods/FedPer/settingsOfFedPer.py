import numpy as np
import csv

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

class FormulationSettings(object):
    def __init__(self, givenDatasetObj):
        self.__N, self.__d = givenDatasetObj.getNumberOfSamples(), givenDatasetObj.getNumberOfFeatures()
        self.__featureIdListOfSharingComponent = []
        self.__featureIdListOfPersonalizedComponent = range(0, self.__d)
        self.__d1, self.__d2 = len(self.__featureIdListOfSharingComponent ), len(self.__featureIdListOfPersonalizedComponent)        
        self.__MMat = MMatrix(givenFeatureIdListOfSharingComponent=self.__featureIdListOfSharingComponent, givenNumOfFeatures=self.__d).get()
        self.__NMat = NMatrix(givenFeatureIdListOfPersonalizedComponent=self.__featureIdListOfPersonalizedComponent, givenNumOfFeatures=self.__d).get()

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
    pass













