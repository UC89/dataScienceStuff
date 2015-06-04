import matplotlib.pyplot as plt
import pandas
import numpy as np



def customPlot(dataIn,binLabels=10,titleIn='Histogram',yLabelIn='frequency'):
	"""
	Takes in an array of dataframe columns to plot
	"""
	setNumber=0
	for dSet in dataIn:
		dSet.hist(bins=binLabels,label=['Data: '+str(setNumber)])
		setNumber+=1
	plt.title(titleIn)
	plt.xlabel('Unset')
	plt.ylabel(yLabelIn)
	plt.legend()
	plt.show()