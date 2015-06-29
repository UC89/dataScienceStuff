import pandas
import numpy as np
import scipy.stats
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from ggplot import *
import myModules

def visualizeVariableWeight():
	"""
	Here I would like to make a function that performs a linear
	regression with gradient descent and makes a scattergram with
	circles representing the weight of each variable
	"""

def linear_regression(features, values):
	#added reshape
   	lr  = linear_model.LinearRegression(fit_intercept=True, normalize=True)
   	features_reshaped = features.reshape((features.shape[0],-1))
   	values_reshaped = values.reshape((values.shape[0],-1))
   	lr.fit(features_reshaped,values_reshaped)
   	intercept = lr.intercept_
   	params = lr.coef_
   	return intercept, params

def calculateR2(data,predictions):
	dataMean = np.mean(data)
	dataDiff = np.power((data - dataMean),2)
	predictionDiff = np.power((data - predictions),2)
	SSres = np.sum(predictionDiff)
	SStot = np.sum(dataDiff)
	r_squared = 1-(SSres/SStot)

def plotGraph():
	binLabels  = []
	for x in range(20):
		binLabels.append(x*250)
	plt.title('Histogram of ENTRIESn_hourly')
	plt.xlabel('ENTRIESn_hourly')
	plt.ylabel('Frequency')
	rainyDays['ENTRIESn_hourly'].hist(bins=binLabels,label=['Rain'])
	notRainyDays['ENTRIESn_hourly'].hist(bins=binLabels,label=['No rain'])
	plt.legend()
	plt.show()

#custom plot ([data],binLabels=10) #preliminary working
def customPlot(dataIn,binLabels=10,titleIn='Histogram',yLabelIn='frequency'):
	setNumber=0
	for dSet in dataIn:
		dSet.hist(bins=binLabels,label=['Data: '+str(setNumber)])
		setNumber+=1
	plt.title(titleIn)
	plt.xlabel('Unset')
	plt.ylabel(yLabelIn)
	plt.legend()
	plt.show()

def compute_r_squared(data, predictions):
	def square(x):
		return x**2
	squareArray = np.vectorize(square)
	dataAverage = np.mean(data)
	ssRes = np.sum(squareArray(data-predictions))
	ssTot = np.sum(squareArray(data-dataAverage))
	r_squared = 1-(ssRes/ssTot)
	return r_squared

def normalize_features(features):
	'''
	Returns means and standard deviations of feature setNumber
	also returns the normalized feature set array
	'''
	means = np.mean(features,axis=0)
	std_devs = np.std(features, axis=0)
	normalized_features = (features - means) / std_devs
	return means, std_devs, normalized_features

def recover_params(means, std_devs, norm_intercept, norm_params):
	intercept = norm_intercept - np.sum(means *norm_params / std_devs)
	params = norm_params / std_devs
	return intercept, params

def bestLinearFits(dataIn,testVar):  #To find best values for best linear regression. TestVar should be string. Returns R2
	'''
		returns linearFits
		Dictionary key = columnName
		Values = feature_values, intercept, params, predictionValue
	'''
	actualValues = dataIn[testVar].values
	linearFits = {}
	#dummy_units = pandas.get_dummies(dataIn['UNIT'],prefix='unit')
	#dataIn = dataIn.join(dummy_units)
	for columnName in dataIn.columns:
		if columnName == testVar or type(dataIn[columnName].values[1])==str:
			print 'This is the test Var'
		else:
			feature_values = dataIn[columnName].values
			intercept,params = linear_regression(feature_values,actualValues)
			lrResult = intercept + np.dot(feature_values,params)
			#linearFits[columnName]=lrResult
			print 'Feature Values Shape: ',feature_values.shape
			print 'Actual Value Shape: ',actualValues.shape
			linearFits[columnName]=[feature_values,intercept,params,lrResult]
	return linearFits


def prelimInvestigate(dataIn):
	print dataIn.describe()
	for columnName in dataIn.columns:
		dataType = dataIn[columnName].dtype
		print columnName, "Type: ",dataType
		if dataType == 'int64' or dataType == 'float64':
			print '\tMean: ',dataIn[columnName].mean(), ' Max: ',dataIn[columnName].max(), ' Min: ',dataIn[columnName].min(), ' Range: ', (dataIn[columnName].max()-dataIn[columnName].min()), 'StDev: ', dataIn[columnName].std()
		else:
			print columnName,' Other type'

def seeData(dataIn):
	print dataIn.head(10)
	print dataIn['DATEn'].head(2)
	print '1',dataIn['DATEn'][0]
	print '2',dataIn['DATEn'][2]





data_raw = pandas.read_csv('turnstile_data_master_with_weather.csv')

def random():
	data_raw['daily_riders'] = data_raw.ENTRIESn_hourly
	rainyDays = data_raw[data_raw['rain'] == 0]
	print 'There are ',rainyDays['rain'].count(),' rainy days.'
	print '\t there is an average of ',rainyDays['ENTRIESn_hourly'].mean(),' Riders'
	notRainyDays = data_raw[data_raw['rain']== 1]
	print 'There are ',notRainyDays['rain'].count(),' days with NO rain'
	print '\t there is an average of ',notRainyDays['ENTRIESn_hourly'].mean(), ' Riders'

	u = scipy.stats.mannwhitneyu(rainyDays['ENTRIESn_hourly'],notRainyDays['ENTRIESn_hourly'])
	print 'U: ',u

	print 'rain: ',data_raw['rain'].count()
	print 'thunder: ',data_raw['thunder'].count()
	print 'daily_riders: ', data_raw['daily_riders'].count()
	features = data_raw[['rain','thunder','meantempi','Hour']]
	values = data_raw['ENTRIESn_hourly']
	features_array = features.values
	values_array = values.values

	lrResult = linear_regression(features,values)
	intercept,params = linear_regression(features_array,values_array)
	print lrResult
	print "Intercepts: ", intercept
	print "Params: ", params
	predictions = intercept + np.dot(features_array,params)
	#y_predicted  = predictions[1]
	#print 'Actual: ',values
	difference  = values_array-predictions
	#difference.hist()
	#plt.show()

	print 'r2',r2_score(values_array,predictions)
	print 'r2 Calculated: ',calculateR2(values_array,predictions)



daysOfWeek = addDayOfWeekColumn(data_raw)
groupByDayOfWeek = data_raw.groupby('dayOfWeek')
averageRidersDayOfWeek = groupByDayOfWeek['ENTRIESn_hourly'].aggregate(np.mean).reset_index()

#prelimInvestigate(data_raw)
#seeData(data_raw)
linearFits = bestLinearFits(data_raw,'ENTRIESn_hourly')
print linearFits




#print plot_1

