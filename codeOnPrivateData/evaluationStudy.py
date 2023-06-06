import numpy as np
import os, sys
from datasets import ToyDataOfConvexCluster, Ijcnn1OfLibsvm, LngxbhbcdexzlhzcxfxycDataset, LngxbhbwexzlhzcxfxycDataset, IITnbswmbbfxycDataset, IrisOfLibsvm, HalfMoonsDataset, BreastCancerOfLibsvm, Covid19EventPredictionDataset, AkiPredictionDataset
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/APFL'))
from APFL import Evaluater as EvaluaterOfAPFL
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/FedAmp'))
from FedAMP import Evaluater as EvaluaterOfFedAMP
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/FedAvg'))
from FedAvg import Evaluater as EvaluaterOfFedAvg
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/FedPer'))
from FedPer import Evaluater as EvaluaterOfFedPer
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/FedProx'))
from FedProx import Evaluater as EvaluaterOfFedProx
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/FPFC'))
from FPFC import Evaluater as EvaluaterOfFPFC
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/pFedMe'))
from pFedMe import Evaluater as EvaluaterOfpFedMe
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/Ditto'))
from Ditto import Evaluater as EvaluaterOfDitto
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/IFCA'))
from IFCA import Evaluater as EvaluaterOfIFCA
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/L2GD'))
from L2GD import Evaluater as EvaluaterOfL2GD
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/pFedMe'))
from pFedMe import Evaluater as EvaluaterOfpFedMe
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'comparedMethods/SuPerFed'))
from SuPerFed import Evaluater as EvaluaterOfSuPerFed
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'pFedNet'))
from pFedNet import Evaluater as EvaluaterOfpFedNet
             
class Dataset(object):
    def __init__(self, givenName):
        self.__name = givenName
        self.__dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        pass
    
    def get(self):
        if self.__name == 'half moon':
            datasetObj = HalfMoonsDataset(givenFn = os.path.join(self.__dir, 'data', 'others', 'clustering', 'halfMoons.csv'))
            return datasetObj
        if self.__name == 'Ijcnn1OfLibsvm':
            datasetObj = Ijcnn1OfLibsvm(givenFn = '/home/yawei/data/libsvmDatasets/classification/ijcnn1.txt')
            return datasetObj
        if self.__name == 'BreastCancerOfLibsvm':
            datasetObj = BreastCancerOfLibsvm(givenFn = os.path.join(self.__dir, 'data', 'libsvmDatasets', 'classification', 'breastCancer.txt'))
            return datasetObj
        if self.__name == 'Lngxbhbcdexzlhzcxfxyc':
            datasetObj = LngxbhbcdexzlhzcxfxycDataset(givenFn = os.path.join(self.__dir, 'data','medicalDatasets','real','classification', 'Lngxbhbcdexzlhzcxfxyc.csv'))
            return datasetObj
        if self.__name == 'IITnbswmbbfxycDataset':
            datasetObj = IITnbswmbbfxycDataset(givenFn = os.path.join(self.__dir, 'data','medicalDatasets','real','classification', 'IITnbswmbbfxycDataset.csv'))
            return datasetObj
        if self.__name == 'Covid19EventPrediction': 
            datasetObj = Covid19EventPredictionDataset(givenFn = os.path.join(self.__dir, 'data','medicalDatasets','real','classification', 'Covid19EventPrediction.csv'))
            return datasetObj
        if self.__name == 'AkiPrediction':
            datasetObj = AkiPredictionDataset(givenFn = os.path.join(self.__dir, 'data','medicalDatasets','real','classification', 'AkiPrediction.csv'))
            return datasetObj
        pass
    pass

class Evaluation(object):
    def __init__(self, givenNameOfDataset):
        self.__datasetObj = Dataset(givenName=givenNameOfDataset).get()     
        pass
        
    def evaluateOne(self, givenName):
        print("--Evaluate {%s}--" % givenName)
        if givenName == 'pFedNet':
            EvaluaterOfpFedNet(givenDatasetObj = self.__datasetObj).execute()
            return 
        if givenName == 'FedAMP':
            EvaluaterOfFedAMP(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'FedAvg':
            EvaluaterOfFedAvg(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'FedProx':
            EvaluaterOfFedProx(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'FPFC':
            EvaluaterOfFPFC(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'pFedMe':
            EvaluaterOfpFedMe(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'Ditto':
            EvaluaterOfDitto(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'FedPer':
            EvaluaterOfFedPer(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'IFCA':
            EvaluaterOfIFCA(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'APFL':
            EvaluaterOfAPFL(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'L2GD':
            EvaluaterOfL2GD(givenDatasetObj = self.__datasetObj).execute()
            return
        if givenName == 'SuPerFed':
            EvaluaterOfSuPerFed(givenDatasetObj = self.__datasetObj).execute()
            return
    
    def evaluateAll(self):
        print(">>>>>>>>>>>>>>>>>>>Evaluate {%s}<<<<<<<<<<<<<<<<<<<" % self.__datasetObj.getName())
        self.evaluateOne(givenName = 'FedAvg')
        self.evaluateOne(givenName = 'FedProx')
        self.evaluateOne(givenName = 'FPFC')
        self.evaluateOne(givenName = 'pFedMe')
        self.evaluateOne(givenName = 'Ditto')
        self.evaluateOne(givenName = 'FedPer')
        self.evaluateOne(givenName = 'IFCA')
        self.evaluateOne(givenName = 'APFL')
        self.evaluateOne(givenName = 'L2GD')
        self.evaluateOne(givenName = 'SuPerFed')
        self.evaluateOne(givenName = 'pFedNet')
        self.evaluateOne(givenName = 'FedAMP')
    pass




if __name__ == '__main__':
    np.random.seed(0)
    datasetList = ['Lngxbhbcdexzlhzcxfxyc'] # 'Lngxbhbcdexzlhzcxfxyc', 'IITnbswmbbfxycDataset', 'Covid19EventPrediction', 'AkiPrediction'
    for dataset in datasetList:
        Evaluation(givenNameOfDataset = dataset).evaluateOne(givenName='pFedNet')






