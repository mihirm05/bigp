import numpy as np 
import matplotlib.pyplot as plt 

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

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

def plotFinal(years, countryQuantity, yearsTrain, countryQuantityTrain, yearsTest, countryQuantityTest, yearsPredict, countryQuantityPredict, ylabel, sigma):
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


    plt.fill_between(yearsPredict.flat, (countryQuantityPredict.flat-2*sigma), (countryQuantityPredict.flat+2*sigma), 
                     color='green',alpha=0.5,label='95% confidence interval')
    plt.ylabel(ylabel)
    plt.xlabel('Year')
    plt.legend()

def errorPlot(qty1, error, xlabel, ylabel):
    plt.plot(qty1[::-1],error.T)
    plt.plot(qty1[::-1],np.zeros((len(qty1),1)),'k--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

