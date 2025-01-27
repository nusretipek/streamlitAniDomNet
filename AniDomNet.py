#!/usr/bin/env python3

# Import Libraries
import pandas as pd
import numpy as np
import time
import glob
import json
import sys
import csv
import os
import ctypes
from ctypes import *
import scipy
from scipy.optimize import minimize
from multiprocessing import Pool
import warnings
import logging
import argparse
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.set_printoptions(suppress = True)
print(np.__version__, scipy.__version__, pd.__version__)

class DomNetwork:
    def __init__(self, trainDataset, initialLag, fullMatrix, saveResults, powellInitials=None, seed=6668, verbose=True):
        
        # Parameters
        self.initialLag = initialLag
        self.trainDatasetLocation = trainDataset
        self.fullMatrix = fullMatrix
        self.saveResults = saveResults
        self.seed = seed
        self.powellInitials = powellInitials
        self.powellStep = True if self.powellInitials is not None else False
        self.verbose = verbose

        # Parse Data
        self.data = pd.read_csv(self.trainDatasetLocation)[self.initialLag:].reset_index(drop=True)
        self.players = list(set(np.concatenate(self.data.values)))
        self.playerCount = len(self.players)
        self.matrixUpperN = int((self.playerCount * (self.playerCount - 1) / 2))
        self.interactions = []

        ## Train dataset
        for _ in range(len(self.data)): 
            initiatorID, receiverID = self.data.loc[_, [self.data.columns[0], self.data.columns[1]]]
            self.interactions.append([initiatorID, receiverID])
        
        ## Downscale to only interacting pairs: fullMatrix -> False
        self.sorted_arr, self.unique_pairs, self.predLength = None, None, None
        if not self.fullMatrix:
            self.sorted_arr = np.sort(self.interactions, axis=1)
            self.unique_pairs = np.sort(np.unique(self.sorted_arr, axis=0))
            self.predLength = len(self.unique_pairs)
            counterLoop, counterPairs, self.indices = 0, 0, []
            for idx in range(self.playerCount):
                for idy in range(idx+1, self.playerCount):
                    if counterPairs < self.predLength and self.unique_pairs[counterPairs][0] == idx and self.unique_pairs[counterPairs][1] == idy:
                        self.indices.append(counterLoop)
                        counterPairs += 1
                    counterLoop += 1
           
        # Convert Interactions to Numpy
        self.interactions = np.array(self.interactions, dtype=np.int32)

        # Create Label Pointers
        self.trainPred = (POINTER(c_int) * len(self.interactions))()
        for i, dim1 in enumerate(self.interactions):
            self.trainPred[i] = (c_int * len(dim1))(*dim1)  
        
        # Set C Function
        self.domnet = CDLL("./anidomnet.so")
        self.domnet.calculateLossCauchy.argtypes = [POINTER(c_double), POINTER(POINTER(c_int)), c_int, c_int]
        self.domnet.calculateLossCauchy.restype = POINTER(c_double)        
        self.domnet.getScores.argtypes = [POINTER(c_double), POINTER(POINTER(c_int)), c_int, c_int]
        self.domnet.getScores.restype = POINTER(POINTER(POINTER(c_double)))
        self.domnet.getErrorIndices.argtypes = [POINTER(c_double), POINTER(POINTER(c_int)), c_int, c_int]
        self.domnet.getErrorIndices.restype = POINTER(c_int) 
        
        # Set initial variables for optimization  -> np.random.uniform(-0.3, 0.3, 4).tolist() #2 * [0.0] + 2 * [0.0] 
        self.initials = None
        self.paramCount = 4
        
        if self.powellInitials is None:
            np.random.seed(self.seed)
            if self.fullMatrix:
                initials = np.random.uniform(-0.1, 0.1, self.matrixUpperN).tolist() + \
                           np.random.uniform(0.0, 0.1, self.paramCount).tolist() 
                self.initials = np.array(initials, dtype=np.float64)

            else:
                initials = np.random.uniform(-0.1, 0.1, self.predLength).tolist() + \
                           np.random.uniform(0.0, 0.1, self.paramCount).tolist() 
                self.initials = np.array(initials, dtype=np.float64)
        else:
            self.initials = self.powellInitials
     
        # Set Bounds for the optimization
        self.boundsMinMax = [451.89080648, 451.89080648, 451.89080648]
        self.bounds = None
        if self.fullMatrix:
            self.bounds = self.matrixUpperN * [(-self.boundsMinMax[0], self.boundsMinMax[0])] + \
                                               4 * [(-self.boundsMinMax[1], self.boundsMinMax[1])]
        else:
            self.bounds = self.predLength * [(-self.boundsMinMax[0], self.boundsMinMax[0])] + \
                                             4 * [(-self.boundsMinMax[1], self.boundsMinMax[1])] 
        # Set iteration variables
        self.time, self.c, self.counterLoss = 0, 0, 0
        self.bestParamsLoss, self.bestLoss, self.bestLossAccuracy = None, None, None
        self.bestParamsAccuracy, self.bestAccuracy, self.bestAccuracyLoss = None, None, None
        self.currentLoss = None
        self.probabilityThreshold = 1 if self.powellStep else 1

    def testSet(self, testFile, arr):
        df = pd.read_csv(testFile)
        counter = 0
        for idx, row in df.iterrows():
            winner_idx = row['Initiator']
            loser_idx = row['Receiver']
            pwin = None
            if winner_idx < loser_idx:
                pwin = 0.5 + np.arctan(arr[winner_idx][loser_idx]) / 3.14159265358979323846
            else:
                pwin = 1 - (0.5 + np.arctan(arr[loser_idx][winner_idx]) / 3.14159265358979323846)
            if pwin >= 0.50:
                counter += 1
        return counter/len(df)
        
    # Early stop criteria, if accuracy 1.0 after 10,000 iterations or loss do not decrease after 10,000 iterations
    def earlyStop(self, x):
        if (self.bestAccuracy == 1 and self.c >= 10000 and self.bestAccuracyLoss <= 1) or self.counterLoss >= 10000 or self.c > 500000:
            raise StopIteration
    
    # Optimization function, call C shared library for calculation
    def CauchyLoss(self, pred): 
        
        # Downscale: fullMatrix -> False
        if self.fullMatrix:
            inputParameters = pred
        else:
            predFull = np.zeros(self.matrixUpperN, dtype=np.float64)
            predFull[self.indices] = pred[:-self.paramCount]
            predFull = np.concatenate((predFull, pred[-self.paramCount:]))
            inputParameters = predFull
    
        # Call C function to calculate probability of wins
        individual_losses = self.domnet.calculateLossCauchy(inputParameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                            self.trainPred, 
                                                            ctypes.c_int(self.playerCount),
                                                            ctypes.c_int(self.interactions.shape[0]))
        
        # Retrieve probabilities & free the memory allocated by the C function
        individual_losses_np = np.fromiter(individual_losses, dtype=np.float64, count=self.interactions.shape[0])
        self.domnet.free(individual_losses)
        
        # Calculate loss & accuracy
        loss = -np.sum(np.log(individual_losses_np))
        accuracy = np.sum(individual_losses_np >= 0.5) / self.interactions.shape[0]

        # Store the best loss and accuracy parameters
        if np.isfinite(loss):
            self.currentLoss = loss
            if self.bestLoss is None or loss <= self.bestLoss:
                self.bestLoss = loss
                self.bestParamsLoss = pred
                self.bestLossAccuracy = accuracy
                self.counterLoss = 0
                if self.saveResults:
                    np.save(name + '_params.npy', self.bestParamsLoss)
            else:
                self.counterLoss += 1
            if self.bestAccuracy is None or accuracy >= self.bestAccuracy:
                self.bestAccuracyLoss = loss
                if self.bestAccuracy is None or accuracy >= self.bestAccuracy:
                    self.bestParamsAccuracy = pred      
                    self.bestAccuracy = accuracy

        # Verbose, every 1000 iterations print out current optimization state
        self.c += 1
        if self.verbose:
            if self.c % 1000 == 0:
                print(str(self.c).ljust(8), 
                      str(round(loss, 6)).ljust(10), 
                      str(round(self.bestLoss, 6)).ljust(10), 
                      str(round(self.bestAccuracy, 4)).ljust(6), 
                      str(round(accuracy, 4)).ljust(6))
                sys.stdout.flush()
        
        # Return statement
        individual_losses_np[individual_losses_np > self.probabilityThreshold] = 1
        loss = -np.sum(np.log(individual_losses_np))
        return loss
    
    def minimizeLossBFGS(self):
        startTime = time.time()
        res0 = None
        
        def constraint_function(x):
            return [x[-2] - 3*x[-1],
		    x[-4] - 3*x[-3]]
        
        constraint = {'type': 'ineq', 'fun': constraint_function}
        
        # Initial SLSQP
        try:
            res0 = minimize(
                self.CauchyLoss,
                self.initials,
                callback=self.earlyStop,
                method='SLSQP',
                options={'maxiter': 20000, 'eps': 1.4901161193847656e-08,},
                tol=1e-20, 
                constraints=constraint,
                bounds=(self.predLength * [(-10000, 10000)] + self.paramCount * [(0, np.inf)]))
            
        except StopIteration:
            if self.verbose:
                print('Early stopping...')  
     
        self.time = time.time() - startTime
        return res0
    
    def minimizeLossPowell(self):
        startTime = time.time()
        res0 = None
       
        try:
            res0 = minimize(self.CauchyLoss, 
                            self.initials,
                            callback=self.earlyStop, 
                            method='Powell', 
                            options={'maxiter': 15000}, 
                            tol= 1e-20,
                            bounds = self.predLength * [(-10000, 10000)] + self.paramCount * [(0, np.inf)])
            
        except StopIteration:
            if self.verbose:
                print('Early stopping...')  
     
        self.time = time.time() - startTime
        return res0
    
    def minimizeLoss(self):
        if self.powellInitials is None:
            self.minimizeLossBFGS()
        else:
            self.minimizeLossPowell()
    
    def getFullScores(self):
        
        # Parse parameters to Numpy
        params = np.array(self.bestParamsAccuracy, dtype=np.float64)
        
        # Downscale: fullMatrix -> False
        if self.fullMatrix:
            inputParameters = params
        else:
            predFull = np.zeros(self.matrixUpperN, dtype=np.float64)
            predFull[self.indices] = params[:-self.paramCount]
            predFull = np.concatenate((predFull, params[-self.paramCount:]))
            inputParameters = predFull
        
        scores = self.domnet.getScores(inputParameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                       self.trainPred, 
                                       ctypes.c_int(self.playerCount),
                                       ctypes.c_int(self.interactions.shape[0]))
        
        interactionScores = np.array([[[scores[i][j][k] for k in range(self.playerCount)] for j in range(self.playerCount)] for i in range(self.interactions.shape[0])])

        # Clear memory
        for i in range(self.interactions.shape[0]):
            for j in range(self.playerCount):
                self.domnet.free(scores[i][j])
            self.domnet.free(scores[i])
        self.domnet.free(scores)
        
        return interactionScores.astype(np.float16)
    
    def getErrorMetrics(self, params=None):
        
        if self.bestParamsAccuracy is None and params is None:
            raise ParameterNotFound
        
        if self.bestParamsAccuracy is None:
            self.bestParamsAccuracy = params
        
        # Parse parameters to Numpy
        params = np.array(self.bestParamsAccuracy, dtype=np.float64)
        
        # Downscale: fullMatrix -> False
        if self.fullMatrix:
            inputParameters = params
        else:
            predFull = np.zeros(self.matrixUpperN, dtype=np.float64)
            predFull[self.indices] = params[:-self.paramCount]
            predFull = np.concatenate((predFull, params[-self.paramCount:]))
            inputParameters = predFull
        
        errors = self.domnet.getErrorIndices(inputParameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                             self.trainPred, 
                                             ctypes.c_int(self.playerCount),
                                             ctypes.c_int(self.interactions.shape[0]))
        
        errorArray = np.fromiter(errors, dtype=np.int64, count=self.interactions.shape[0])
        self.domnet.free(errors)
        
        # Get flagged interactions
        indices = np.where(errorArray == 1)[0]

        # Return indices, total interactions, correct predictions
        return (indices, self.interactions.shape[0], self.interactions.shape[0]-len(indices))
    
    
# Optimize Cauchy
def optimizeBFGS(dataset, seed, initials):
    domModel = DomNetwork(trainDataset=dataset, 
                          initialLag=0,
                          fullMatrix=False,
                          saveResults=False,
                          seed=seed,
                          powellInitials=initials,
                          verbose=False)
    domModel.minimizeLoss()

    scores = domModel.getFullScores()
    params = domModel.bestParamsAccuracy
    #print(str(seed).ljust(5), str(round(domModel.bestAccuracyLoss, 3)).ljust(5), str(round(domModel.bestAccuracy, 4)).ljust(5))#, str(round(testAccuracy, 4)).ljust(5))
    return domModel.bestAccuracyLoss, domModel.bestAccuracy, domModel.initials, params, scores
 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process CSV file and arguments')
    parser.add_argument('-file', type=str, help='CSV file path')
    parser.add_argument('-outfolder', type=str, help='Output folder path')
    parser.add_argument('-n', type=int, help='Number of random initializations (Integer value > 0)')
    parser.add_argument('-ncores', type=int, help='Number of cores (Integer value > 0)')
    return parser.parse_args()

    
if __name__ == '__main__':
    
    args = parse_arguments()
    
    # Assertions
    assert args.file.endswith('.csv'), "Invalid filename or format"
    assert isinstance(args.n, int) and args.n > 0, "Invalid value for -n"
    assert isinstance(args.ncores, int) and args.ncores > 0, "Invalid value for -ncores"
    with open(args.file, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        assert len(header) == 2 and header[0] == 'Initiator' and header[1] == 'Receiver', "Invalid CSV format"
    
    # Parse arguments to variable / Initialize
    startTime = time.time()
    name = args.file.rsplit('/', 1)[1].rsplit('.', 1)[0]
    print(name)
    
    dataFile = args.file
    folderPath = args.outfolder
    if not os.path.exists(folderPath):
        os.mkdir(folderPath)

    arguments = []
    for _ in range(args.n):
        arguments.append((dataFile, _, None))
        
    # Multiprocessing
    with Pool(args.ncores) as p:
        resultArr = p.starmap(optimizeBFGS, arguments)
        
    endTime = time.time()
    
    topResults = []
    resultArr = sorted(resultArr, key=lambda x: x[0])
    resultArr = sorted(resultArr, key=lambda x: x[1], reverse=True)

    for idx, item in enumerate(resultArr):
        if idx < 1:
            np.save(folderPath + '/' + name + '_scores.npy', item[4])
            with open(folderPath + '/' + name +'.txt', 'w') as f:
                f.write(str('Time (seconds)\n--------\n'))
                f.write(str(endTime-startTime) + '\n')
                f.write(str('\nInitial parameter values (random)\n--------\n'))
                json.dump(item[2].tolist(), f)
                f.write('\n')
                f.write(str('\nOptimized parameter values (AniDomNet)\n--------\n'))
                json.dump(item[3].tolist(), f)
                f.write('\n')
                f.write('Loss\n--------\n')
                f.write('\n' + str(item[0]) + '\n')
                f.write(str('\nAccuracy\n--------\n'))
                f.write(str(item[1]) + '\n\nAniDomNet Initalizations (id, loss, accuracy)\n--------\n')
        
        with open(folderPath + '/' + name + '.txt', 'a') as f:
            f.write(str(idx) + ' ' +  str(round(item[0], 6)) + ' ' +  str(round(item[1], 6)) + '\n')



                
