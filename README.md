# BikeBuyer
Challenge 2 Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

demo =pd.read_csv('d:\Darek\AdvWorksCusts.csv')
avs=pd.read_csv('d:\Darek\AW_AveMonthSpend.csv')
bb=pd.read_csv('d:\Darek\AW_BikeBuyer.csv')

import datetime

gvr=datetime.date(1998,1,1)
gvr=pd.to_datetime(gvr)
demo['BirthDate']=pd.to_datetime(demo['BirthDate'])

l=len(demo['BirthDate'])
for n in range(l):
	demo.loc[n,'Age']=gvr.year - demo.loc[n,'BirthDate'].year

for n in range(len(fm)):
	if fm.loc[n,'Age'] < 25:
		if fm.loc[n,'Sex']=='M':
			fm.loc[n,'AgeLabel']='Males under 25 years of age'
		else:
			fm.loc[n,'AgeLabel']='Females under 25 years of age'
	elif fm.loc[n,'Age']>=25:
		if fm.loc[n,'Age']<=45:
			if fm.loc[n,'Sex']=='M':
				fm.loc[n,'AgeLabel']='Males aged between 25 and 45'
			else:
				fm.loc[n,'AgedLabel']='Females aged between 25 and 45'
	elif fm.loc[n,'Age']>55:
		if fm.loc[n,'Sex']=='M':
			    fm.loc[n,'AgeLabel']='Males over 55 years of age'
		else:
			    fm.loc[n,'AgeLabel']='Females over 55 years of age'
	else:
	       fm.loc[n,'AgeLabel']='between 45 and 55'

		   
		   
def plot_box(ap, cols, col_y = 'AveMonthSpend'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col, col_y, data=fm)
        plt.xlabel(col) # Set text for the x axis
        plt.ylabel(col_y)# Set text for y axis
        plt.show()	
		
cat_col=['AgeLabel']

plot_box(fm, cat_cols)


def bikebuyer(col):
       labels=demo[col].unique()
       for lab in labels:
	       count=0
	       for n in range(len(demo)):
		       if demo.loc[n,'BikeBuyer']==1:
			       if demo.loc[n,col]==lab:
				       count=count+1
	       print('Number of Bike Buyers in ' + lab + ' is ' + str(count))

       
>>> bikebuyer('Occupation')

##Classification
print(demo.shape)
##There are 16519 rows and 25 columns in the dataset.
##The first column is 'CustomerID', which is an identifier. We will drop this since this is not a feature.
demo.drop(['CustomerID'], axis=1, inplace=True)
##Usnołem poniższe kolumny, ponieważ nie powinny wpływać na model
demo.drop(['Title'], axis=1, inplace=True)
demo.drop(['FirstName'], axis=1, inplace=True)
demo.drop(['MiddleName'], axis=1, inplace=True)
demo.drop(['LastName'], axis=1, inplace=True)
demo.drop(['Suffix'], axis=1, inplace=True)
demo.drop(['AddressLine1'], axis=1, inplace=True)
demo.drop(['AddressLine2'], axis=1, inplace=True)
demo.drop(['City'], axis=1, inplace=True)
demo.drop(['StateProvinceName'], axis=1, inplace=True)
demo.drop(['CountryRegionName'], axis=1, inplace=True)
demo.drop(['PostalCode'], axis=1, inplace=True)
demo.drop(['PhoneNumber'], axis=1, inplace=True)
##Examine classes and class imbalance
##Fortunately, it is easy to test for class imbalance using a frequency table.
##Execute the code in the cell below to display a frequency table of the classes:
bikebuyer_counts = demo['BikeBuyer'].value_counts()
print(bikebuyer_counts)
## Visualize class separation by numeric features
## The primary goal of visualization for classification problems is to understand which features are useful for class separation.
## In this section, you will start by visualizing the separation quality of numeric features. 
def plot_box(demo, cols, col_x = 'BikeBuyer'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.boxplot(col_x, col, data=demo)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()
num_cols = ['MaritalStatus','HomeOwnerFlag', 'NumberCarsOwned', 'NumberChildrenAtHome' , 'TotalChildren', 'YearlyIncome' ]
plot_box(demo, num_cols)

def plot_violin(demo, cols, col_x = 'BikeBuyer'):
    for col in cols:
        sns.set_style("whitegrid")
        sns.violinplot(col_x, col, data=demo)
        plt.xlabel(col_x) # Set text for the x axis
        plt.ylabel(col)# Set text for y axis
        plt.show()

plot_violin(demo, num_cols)

cat_cols=['AddressLine1','City','StateProvinceName','CountryRegionName','PostalCode','PhoneNumber','BirthDate','Education','Occupation','Gender','MaritalStatus']

cat_cols=['Education','Occupation','Gender','MaritalStatus']

demo['dummy'] = np.ones(shape = demo.shape[0])

for col in cat_cols:
    print(col)
    counts = demo[['dummy', 'BikeBuyer', col]].groupby(['BikeBuyer', col], as_index = False).count()
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    _ = plt.figure(figsize = (10,4))
    plt.subplot(1, 2, 1)
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n NO BikeBuyer')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp = counts[counts['BikeBuyer'] == 1][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n TRUE BikeBuyer')
    plt.ylabel('count')
    plt.show()
##Wykres kupujących w zalezności od wieku
demo['dummy'] = np.ones(shape = demo.shape[0])

cat_cols=['Age']

for col in cat_cols:
    print(col)
    counts = demo[['dummy', 'BikeBuyer', col]].groupby(['BikeBuyer', col], as_index = False).count()
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    _ = plt.figure(figsize = (10,4))
    plt.subplot(1, 2, 1)
    temp = counts[counts['BikeBuyer'] == 0][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n NO BikeBuyer')
    plt.ylabel('count')
    plt.subplot(1, 2, 2)
    temp = counts[counts['BikeBuyer'] == 1][[col, 'dummy']]
    plt.bar(temp[col], temp.dummy)
    plt.xticks(rotation=90)
    plt.title('Counts for ' + col + '\n TRUE BikeBuyer')
    plt.ylabel('count')
    plt.show()

##Zliczanie wystąpień danej etykiety
demo['Education'].value_counts()

##Two of these categories: 'Graduate Degree' and 'High School' have a sililar distribution of BikeBuyer. Execute the code in the cell below to aggregate these categories.
edu_cats = {'Bachelors':'Bachelors', 'Partial College':'Partial College', 'High School':'GraDeg_HigSch', 'Graduate Degree':'GraDeg_HigSch', 'Partial High School':'Partial High School'}

demo['Education'] = [edu_cats[x] for x in demo['Education']]

demo['log_YearlyIncome'] = demo['YearlyIncome'].apply(math.log)

##zapisywanie pliku
demo.to_csv('d:\Darek\demo_14cols_logAge_logYI.csv')
##Basics of logistic regression!!!!!!!!!
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

##Prepare data for scikit-learn model  !!!!
##With the data prepared, it is time to create the numpy arrays required for the scikit-learn model. 
##The code in the cell below creates a numpy array of the label values 'BikeBuyer'
labels = np.array(demo['BikeBuyer'])

##Now, you need to create the numpy feature array or model matrix. As first step, the categorical variables need to be recoded as binary dummy variables. As discussed in another lesson this is a three step process:
## 1. Encode the categorical string variables as integers.
def encode_string(cat_features):
    ## First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()
	
categorical_columns = ['Occupation','Gender','MaritalStatus']

Features = encode_string(demo['Education'])

for col in categorical_columns:
    temp = encode_string(demo[col])
    Features = np.concatenate([Features, temp], axis = 1)
	
print(Features.shape)
print(Features[:2, :])
	
## 2. Transform the integer coded variables to dummy variables.
## 3. Append each dummy coded categorical variable to the model matrix.
##Next the numeric features must be concatenated to the numpy array by executing the code in the cell below. 
Features = np.concatenate([Features, np.array(demo[['HomeOwnerFlag','NumberCarsOwned',
'NumberChildrenAtHome','TotalChildren','log_YearlyIncome','log_Age']])], axis = 1)

print(Features.shape)
print(Features[:2, :])


##You must split the cases into training and test data sets. This step is critical. If machine learning models are tested on the training data, the results will be both biased and overly optimistic.
##The code in the cell below performs the following processing:
## 1. An index vector is Bernoulli sampled using the train_test_split function from the model_selection package of scikit-learn. 
## 2. The first column of the resulting index array contains the indices of the samples for the training cases. 
## 3. The second column of the resulting index array contains the indices of the samples for the test cases. 
## Execute the code. 

import numpy.random as nr

nr.seed(9988)

indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 5000)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])


##There is just one more step in preparing this data. Numeric features must be rescaled so they have a similar range of values. Rescaling prevents features from having an undue influence on model training simply because then have a larger range of numeric variables.
##The code in the cell below uses the StanardScaler function from the Scikit Learn preprocessing package to Zscore scale the numeric features. Notice that the scaler is fit only on the training data. The trained scaler is these applied to the test data. Test data should always be scaled using the parameters from the training data.

scaler = preprocessing.StandardScaler().fit(X_train[:,18:])
X_train[:,18:] = scaler.transform(X_train[:,18:])							  
X_test[:,18:] = scaler.transform(X_test[:,18:])
print(X_train[:2,])

##Construct the logistic regression model
##Now, it is time to compute the logistic regression model. The code in the cell below does the following:
##Define a logistic regression model object using the LogisticRegression method from the scikit-learn linear_model package.
##Fit the linear model using the numpy arrays of the features and the labels for the training data set.
##Execute this code. 

logistic_mod = linear_model.LogisticRegression()

logistic_mod.fit(X_train, y_train)

##LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
##         intercept_scaling=1, max_iter=100, multi_class='warn',
##          n_jobs=None, penalty='l2', random_state=None, solver='warn',
##          tol=0.0001, verbose=0, warm_start=False)

##dla test 5000
print(logistic_mod.intercept_)						 
##[-0.70811602]

print(logistic_mod.coef_)
##[[ 0.16227359 -0.02271664 -0.24104988 -0.18670795 -0.41991514 -0.16686306
##  -0.77005498  0.4444422  -0.13005423 -0.08558596 -0.70456291 -0.00355311
##  -1.14944367  0.44132764  0.19201117 -0.07838422  0.7864223  -0.01826574
##   0.70098226 -0.2428087 ]]

##dla test 2000
print(logistic_mod.intercept_)
##[-0.70932978]
print(logistic_mod.coef_)
##[[ 0.19440513 -0.01678749 -0.3007383  -0.16331702 -0.42289211 -0.13784425
##  -0.82511674  0.51628036 -0.17476369 -0.08788546 -0.68798376 -0.02134603
##  -1.14365991  0.43433013  0.14329807 -0.06156056  0.79190511 -0.0191918
##   0.72180429 -0.24556718]]

probabilities = logistic_mod.predict_proba(X_test)

print(probabilities[:7,:])
##[[0.70340259 0.29659741]
##[0.90576416 0.09423584]
##[0.90723455 0.09276545]
##[0.92595368 0.07404632]
##[0.34888135 0.65111865]
##[0.63365214 0.36634786]
##[0.65036982 0.34963018]]


print(np.array(scores[:15]))
##[0 0 0 0 1 0 0 0 1 0 0 1 0 0 0]
print(y_test[:15])
##[1 0 0 0 1 0 0 0 0 0 0 0 0 0 0]

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])

print_metrics(y_test, scores)

##                 Confusion matrix
##             Score positive    Score negative
##Actual positive      2975               356
##Actual negative       737               932

##Accuracy  0.78
 
##                Positive      Negative
##Num case     3331          1669
##Precision      0.80            0.72
##Recall           0.89            0.56
##F1                0.84             0.63




print_metrics(demo['BikeBuyer'], scores1)
##                Confusion matrix
##                 Score positive    Score negative
##Actual positive      9848              1182
##Actual negative      2388              3101

##Accuracy  0.78
 
##          Positive      Negative
##Num case    11030          5489
##Precision    0.80          0.72
##Recall       0.89          0.56
##F1           0.85          0.63
