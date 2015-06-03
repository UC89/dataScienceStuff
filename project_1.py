import pandas
import numpy as np
import scipy.stats
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score



def linear_regression(features, values):
   	lr  = linear_model.LinearRegression(fit_intercept=True)
   	lr.fit(features,values)
   	intercept = lr.intercept_
   	params = lr.coef_
   	return intercept, params

data_raw = pandas.read_csv('turnstile_data_master_with_weather.csv')
data_raw['daily_riders'] = data_raw.ENTRIESn_hourly
#lh_batting_averages = batter_data[batter_data['handedness'] == 'L']
#turnstile_weather['column_to_graph'].hist()
print data_raw.describe()
rainyDays = data_raw[data_raw['rain'] == 0]
print 'There are ',rainyDays['rain'].count(),' rainy days.'
print '\t there is an average of ',rainyDays['ENTRIESn_hourly'].mean(),' Riders'
notRainyDays = data_raw[data_raw['rain']== 1]
print 'There are ',notRainyDays['rain'].count(),' days with NO rain'
print '\t there is an average of ',notRainyDays['ENTRIESn_hourly'].mean(), ' Riders'
binLabels  = []
for x in range(20):
	binLabels.append(x*250)
plt.title('Histogram of ENTRIESn_hourly')
plt.xlabel('ENTRIESn_hourly')
plt.ylabel('Frequency')
rainyDays['ENTRIESn_hourly'].hist(bins=binLabels,label=['Rain'])
notRainyDays['ENTRIESn_hourly'].hist(bins=binLabels,label=['No rain'])
#rainyDays.plot(kind='hist')
plt.legend()
#plt.show()

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
#print 'predictted: ',predictions
#print 'Actual: ',values

print 'r2',r2_score(values_array,predictions)
