# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:06:55 2023

@author: anast
"""


import pandas as pd #TBC if needed or just use the next one
import numpy as np
from matplotlib import pyplot as plt

#####Instructions
#A european bank wants to identify the main factors contributing to customer 
#churn (i.e., customers leaving the bank and closing their accounts). 
#By doing so, the bank can target such customers with incentives, or use this 
#knowledge to propose new products that are better suited to their needs.



##### Import the data 

#A: May be a strange way of doing it, but it works very well...!
churn_data =  pd.read_csv('churn_bank.txt', delimiter=',')
churn_data = churn_data.drop(["CustomerId", "Surname"], axis = 1)

churn_data.isnull().sum()
#No missing values

##### Getting to know the dataset

#A: to do --> Use seaborn to have relevant and pretty graphs for each variable
pd.DataFrame.head(churn_data)
col_names = churn_data.columns.values.tolist()
#churn_data.columns() This is an index object, not a list, so can't just print it

len(churn_data) #10 000 customers
len(col_names) #13 features

#Review of each of the variables: 
#A: /!\ Probably a more efficient way of doing this (esp the codes for the frequency!!)
#A: Also see which ones are really needed...

#Surname
np.unique(churn_data.Surname)
len(np.unique(churn_data.Surname)) #2932 surprisingly low!

#Credit score
plt.hist(churn_data.CreditScore)
np.mean(churn_data.CreditScore)
np.median(churn_data.CreditScore)

#Geography
churn_data.Geography

#A: Should we make a dictionnary to show that we know what that is?
countries = np.unique(churn_data.Geography)
freq_country = [sum(churn_data.Geography == i) for i in countries]

plt.bar(countries,freq_country)

#A: Is there a way to show a map of Europe with this? ahahha

#Gender
gender = np.unique(churn_data.Gender)
freq_gender = [sum(churn_data.Gender == i ) for i in gender]
plt.pie(freq_gender, labels = gender)

#Age
plt.hist(churn_data.Age)
np.mean(churn_data.Age)
np.median(churn_data.Age)

#Tenure
plt.hist(churn_data.Tenure) #A: Not very illustrative...
    
#Balance
plt.hist(churn_data.Balance) #A: Large number with very low

#NumOfProducts
plt.hist(churn_data.NumOfProducts) #A: Not illustrative

#HasCrCard
label_HasCrCard = ['Has a credit card', 'Does not have a credit card']
freq_HasCrCard= [sum(churn_data.HasCrCard), 10000-sum(churn_data.HasCrCard) ] #A: OPTIMIZE!!!!!
plt.pie(freq_HasCrCard, labels = label_HasCrCard)

#IsActiveMember
label_IsActiveMember = ['Active member', 'Inactive member']
freq_IsActiveMember= [sum(churn_data.IsActiveMember), 10000-sum(churn_data.IsActiveMember) ] #A: OPTIMIZE!!!!!
plt.pie(freq_IsActiveMember, labels = label_IsActiveMember)

#EstimatedSalary
plt.hist(churn_data.EstimatedSalary) #A: very ugly

##### What is the churn rate for the bank customers?

#the annual percentage rate at which customers leave
churn_rate = sum(churn_data.Exited)/len(churn_data)
#20%, this seems ridiculously high tbh

##### What is the relationships between variables ? 
#Replacing gender by numerical values 
churn_data['Gender'] = churn_data['Gender'].replace({'Female': 1, 'Male': 0})

#simple covariance matrix
correlation_matrix = churn_data.corr()
print(correlation_matrix) 
#In general what we can remark is that explicative variables are not that much correlated with one another, 
#which is good thing since it implies more explicative power for the churn rate.

### For the age variable 

#Activity 
sns.lineplot(x='Age', y='IsActiveMember', data=churn_data, marker='o')

##For all other variables, the mean stays constant, for Balance and Estimated Salary it is 
###What is the distribution of variables across countries ?
#Clients count
country_count = churn_data.groupby('Geography').size().reset_index(name='Count')
custom_colors = ['blue', 'green', 'orange']

plt.bar(grouped_data['Geography'], grouped_data['Count'], color = custom_colors)
plt.ylabel('Number of clients')

#Distributions
grouped_data = churn_data.groupby('Geography').mean().reset_index()

sns.boxplot(y='CreditScore', x='Geography', hue='Geography', data=churn_data).set(xlabel='')
plt.gca().legend().remove()

sns.boxplot(y='Tenure',x = 'Geography', hue = 'Geography', data = churn_data).set(xlabel='')
plt.gca().legend().remove()

sns.boxplot(y='Balance',x = 'Geography', hue = 'Geography',data = churn_data).set(xlabel='')
plt.gca().legend().remove()

sns.boxplot(y='EstimatedSalary',x = 'Geography', hue = 'Geography',data = churn_data).set(xlabel='')
plt.gca().legend().remove()


### Influence of other variables on churn 

sns.countplot(data=churn_data, x='Exited', hue='Gender')
sns.countplot(data=churn_data, x='Exited', hue='Geography')

#Higher churn rate for female compared to male
#Higher churn rate in Germany and spain compared to france, maybe a link with the number of clients

#Are they significative difference ? 
from scipy.stats import ttest_ind

#Separate the churn variable according to gender and geography
female_data = df[df['Gender'] == 1]['Exited']
male_data = df[df['Gender'] == 0]['Exited']

france_data= df [df['Geography'] == 'France']['Exited']
spain_data= df [df['Geography'] == 'Spain']['Exited']
germany_data= df [df['Geography'] == 'Germany']['Exited']

#For gender
t_statistic, p_value = ttest_ind(female_data, male_data)
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')

#For geography 
t_statistic, p_value = ttest_ind(france_data, spain_data)
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')
#Not significant

t_statistic, p_value = ttest_ind(france_data, germany_data)
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')
#p<0.001

t_statistic, p_value = ttest_ind(spain_data, germany_data)
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')
#p<0.001


### Machine learning Model ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Without the country variable : 
Y = df["Exited"]
X = df.drop("Exited", axis = 1)
X = X.drop("Geography", axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)

mse_values= []

for i in range(2, 50): 
    model = RandomForestClassifier(min_samples_split=i, random_state = 42)
    model.fit(X_train, Y_train)
    prediction_Y = model.predict(X_test)
    mse = mean_squared_error(Y_test, prediction_Y)
    mse_values.append(mse)

#Best value for min_samples_split is 42, Accuracy = 0.86

#Ideas for the rest: 
# Q° : Look at interactions between groups: What are the age, salary, balance, number of products, etc. distributions for each gender group?
    #/!\ classify the variables into 2: people that make up groups (gender, age, etc) and indicators (salary, number of products, etc)
    #Use tapply to compare mean zB in different groups --> statistical tests to chec for significativity
# How are the different indicators distributed by country?
    #Seems rather straightforward...
# How do the different variables affect churn? What are the causes that can lead to increased (or reduced) customer churn?
    #Do a linear model to look at how churn is affected by each?
    #We won't be able to know which variables increase or reduce customer churn, but we can see which variables are correlated with increased or decreased churn
#Build a simple machine learning classification model that predicts churn based on customer's features
    #Def doable,cf Mach learning
    # Hardest q° will be which model to use... (RF classification imo, but need to consider question and justify choice)

#Suggestion for ml algorithm, maybe use the end of the last exercise done in class if we don't want to bother, I think it did pretty good on accuracy.

#Open questions:
    #What important points made in class need to be included?
    #Quantitative variables: histograms and boxplots
    #Qualitative variables: bar graphs and pie charts
    #Statistical tests? At least if we do a linear model...
    #ANOVA model to look at the variation of the variance?and interactions?
