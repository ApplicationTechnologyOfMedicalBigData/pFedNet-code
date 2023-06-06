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

class FormulationSettings(object):
    def __init__(self, givenDatasetObj, givenLambda):
        self.__d, self.__N = givenDatasetObj.getNumberOfFeatures(), givenDatasetObj.getNumberOfSamples()
        self.__lambda = givenLambda

    def getNumberOfSamples(self):
        return self.__N
    
    def getNumberOfFeatures(self):
        return self.__d

    def getLambda(self):
        return self.__lambda
    pass


