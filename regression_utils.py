import numpy as np 
import matplotlib.pyplot as plt 
import random 
from itertools import cycle

#extract dataframes corresponding to countries
def countryDF(country, dataframe):
    countryData = dataframe[dataframe['Country'] == country]
    return countryData


#extract columns
def columnExtractor(dataframe, columnName):
    values = dataframe[columnName]
    return values 


# plot values for dataset columns    
def plotQuantities(qty1, qty2, xlabel, ylabel, label, title):
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.scatter(qty1, qty2, label=label)
    plt.title(title)
    plt.legend()
    plt.show()


# Plot the function, the prediction and the 95% confidence interval based on the MSE
def plotFinal(years, countryQuantity, yearsTrain, countryQuantityTrain, yearsTest, countryQuantityTest, yearsPredict, countryQuantityPredict, ylabel, sigma, regression_type):
    plt.figure()
    #plt.plot(x, Y, 'r--',label=r'$f(x) = x\,\sin(x)$')
    #actual data 
    plt.scatter(years, countryQuantity,label='Observations')

    #estimate
    plt.plot(yearsPredict, countryQuantityPredict, 'r--', label='Prediction')
    plt.scatter(yearsTest,countryQuantityTest,label='Missing values')

    #plt.fill(np.concatenate([x, x[::-1]]),
    #         np.concatenate([y_pred - 1.9600 * sigma,
    #                        (y_pred + 1.9600 * sigma)[::-1]])[:,6],
    #         alpha=1, fc='b', ec='None', label='95% confidence interval')
    
    if regression_type == 'Gaussian': 
        plt.fill_between(yearsPredict.flat, (countryQuantityPredict.flat-2*sigma), (countryQuantityPredict.flat+2*sigma), 
                     color='green',alpha=0.5,label='95% confidence interval')
    plt.ylabel(ylabel)
    plt.xlabel('Year')
    plt.legend()


def errorComputation(countryDF, countryQuantityPredict, quantity,regression_type):
    #countryQuantityPredict = countryQuantityPredict[::-1]
    countryQuantityActual = columnExtractor(countryDF,str(quantity)).tolist()
    countryQuantityActual = countryQuantityActual[::-1]
    print(regression_type,'Prediction \n', countryQuantityPredict.T)
    print('Actual \n', countryQuantityActual)
    error = (np.absolute((countryQuantityPredict.T - countryQuantityActual))/countryQuantityActual)*100
    return error
    

#error calculation between predicted value and ground truth
def errorPlot(qty1, error, xlabel, ylabel,regression_type,color):
    
    plt.plot(qty1[::-1], np.ones((len(qty1),1))*np.mean(error.T), '--', c=color, label=regression_type+' mean')
    print('mean absolute percentage error',regression_type,': ',np.mean(error.T))
    plt.plot(qty1[::-1],error.T, '-', c=color, label=regression_type)
    #plt.plot(qty1[::-1],np.zeros((len(qty1),1)),'k--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()


#matrix randomizer
def randomizer(countryQuantity, years, split):
    countryQuantity = countryQuantity.tolist()
    years = years.tolist()
     
    print('Train:Test split is: ',split,':',16-split)

    #combine both the lists and randomize while maintaining the mapping 
    combinedZip = list(zip(years,countryQuantity))
    random.shuffle(combinedZip) 

    #unzip the combination 
    years,countryQuantity = zip(*combinedZip) 

    countryQuantity = list(countryQuantity)
    countryQuantityTrain = countryQuantity[:split]

    years = list(years)
    yearsTrain = years[:split]

    countryQuantityTest = countryQuantity[split:]
    yearsTest = years[split:]

    #countryQuantity = countryQuantityTrain 
    #years = yearsTrain

    countryQuantityTrain = np.asarray(countryQuantityTrain).reshape(-1,1)
    yearsTrain = np.asarray(yearsTrain).reshape(-1,1)

    countryQuantityTest = np.asarray(countryQuantityTest).reshape(-1,1)
    yearsTest = np.asarray(yearsTest).reshape(-1,1)

    return countryQuantityTrain, yearsTrain, countryQuantityTest, yearsTest
    
    
    
    
        

