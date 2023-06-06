import os

class ModelLogger(object):
    def __init__(self, givenSpecificModel, givenTargetPath = os.path.dirname(os.path.realpath(__file__))):
        self.__specificModel, self.__targetPath = givenSpecificModel, givenTargetPath
        self.__testClassicAccList = self.__specificModel.getTestClassicAccuracyRecords(givenType = 'str')
        self.__testBalancedAccList = self.__specificModel.getTestBalancedAccuracyRecords(givenType = 'str')
        self.__trainingClassicAccList = self.__specificModel.getTrainingClassicAccuracyRecords(givenType = 'str')
        self.__trainingBalancedAccList = self.__specificModel.getTrainingBalancedAccuracyRecords(givenType = 'str')
        self.__testLossList = self.__specificModel.getTestLossRecords(givenType = 'str')
        self.__trainingLossList = self.__specificModel.getTrainingLossRecords(givenType = 'str')
        self.__pModelStr = self.__specificModel.getPersonalizedModel(givenType = 'str')
        pass
    
    def __initTargetFilePath(self, givenFilePath):
        if os.path.exists(givenFilePath) == False:
            dir = os.path.dirname(givenFilePath)
            if os.path.exists(dir) == False:
                os.makedirs(dir)
            with open(givenFilePath, mode='a', encoding='utf-8') as f:
                print('Successfully creat {%s}' % givenFilePath)
        pass
                        
    def __toFile(self, givenFullPath, givenStr):
        self.__initTargetFilePath(givenFilePath=givenFullPath)
        f = open(givenFullPath, 'w')
        self.__print(givenMsg = givenStr, givenFileObject=f)
        f.close()
    
    def logTrainingLoss(self, givenDatasetName, givenNumOfClients, givenMethod):
        fullPath = os.path.join(self.__targetPath, 'output', givenDatasetName, str(givenNumOfClients)+' clients', givenMethod, 'trainingLoss.log')
        self.__toFile(givenFullPath=fullPath, givenStr=self.__trainingLossList)
        pass
    
    def logTestLoss(self, givenDatasetName, givenNumOfClients, givenMethod):
        fullPath = os.path.join(self.__targetPath, 'output', givenDatasetName, str(givenNumOfClients)+' clients', givenMethod, 'testLoss.log')
        self.__toFile(givenFullPath=fullPath, givenStr=self.__testLossList)
        pass
    
    def __logAccuracyUnit(self, givenDir, givenType):
        if givenType == 'test classic accuracy':
            fullPath = os.path.join(givenDir, 'testClassicAcc.log')
            self.__toFile(givenFullPath=fullPath, givenStr=self.__testClassicAccList)
        if givenType == 'test balanced accuracy':
            fullPath = os.path.join(givenDir, 'testBalancedAcc.log')
            self.__toFile(givenFullPath=fullPath, givenStr=self.__testBalancedAccList)
        if givenType == 'training classic accuracy':
            fullPath = os.path.join(givenDir, 'trainingClassicAcc.log')
            self.__toFile(givenFullPath=fullPath, givenStr=self.__trainingClassicAccList)
        if givenType == 'training balanced accuracy':
            fullPath = os.path.join(givenDir, 'trainingBalancedAcc.log')
            self.__toFile(givenFullPath=fullPath, givenStr=self.__trainingBalancedAccList)
        pass
    
    def logTrainingAccuracy(self, givenDatasetName, givenNumOfClients, givenMethod):
        dir = os.path.join(self.__targetPath, 'output', givenDatasetName, str(givenNumOfClients)+' clients', givenMethod)
        self.__logAccuracyUnit(givenDir=dir, givenType='training classic accuracy')
        self.__logAccuracyUnit(givenDir=dir, givenType='training balanced accuracy')
        pass
    
    def logTestAccuracy(self, givenDatasetName, givenNumOfClients, givenMethod):
        dir = os.path.join(self.__targetPath, 'output', givenDatasetName, str(givenNumOfClients)+' clients', givenMethod)
        self.__logAccuracyUnit(givenDir=dir, givenType='test classic accuracy')
        self.__logAccuracyUnit(givenDir=dir, givenType='test balanced accuracy')
        pass

    def logFinalModel(self, givenDatasetName, givenNumOfClients, givenMethod):
        fullPath = os.path.join(self.__targetPath, 'output', givenDatasetName, str(givenNumOfClients)+' clients', givenMethod, 'testFinalModel.log')
        self.__toFile(givenFullPath=fullPath, givenStr = self.__pModelStr)
        pass
    
    def toTerminal(self, givenStr):
        self.__print(givenStr)
    
    def __print(self, givenMsg, givenFileObject = 'None'):
        if givenFileObject != 'None': # if a file is opened.
            print(givenMsg, file = givenFileObject)
            return 
        print(givenMsg)
    
    def setTargetPath(self, givenTargetPath):
        self.__targetPath = givenTargetPath
    pass
    














