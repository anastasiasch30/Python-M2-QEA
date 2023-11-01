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
freq_country = [ sum(churn_data.Geography == i) for i in countries]

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

#Open questions:
    #What important points made in class need to be included?
    #Quantitative variables: histograms and boxplots
    #Qualitative variables: bar graphs and pie charts
    #Statistical tests? At least if we do a linear model...
    #ANOVA model to look at the variation of the variance?and interactions?