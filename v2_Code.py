# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 21:06:55 2023

@author: anast
"""
#NB need prints somewhere!!
#Save all relevant graphs

#Harmoniser les " et ', mettre que des " car ' est utilisé en grammaire anglaise

import pandas as pd #TBC if needed or just use the next one
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

#####Instructions
#A european bank wants to identify the main factors contributing to customer 
#churn (i.e., customers leaving the bank and closing their accounts). 
#By doing so, the bank can target such customers with incentives, or use this 
#knowledge to propose new products that are better suited to their needs.


##### Import the data 

#A: May be a strange way of doing it, but it works very well...!
churn_data =  pd.read_csv('churn_bank.txt', delimiter=',')
#churn_data = churn_data.drop(["CustomerId", "Surname"], axis = 1) 
#Attention: pourquoi est-ce qu'on drop CustomerId et Surname? Nécessaire pourtant

churn_data.isnull().sum()
#No missing values




##### Getting to know the dataset

print(churn_data.head())

col_names = churn_data.columns.values.tolist()
print("There are ",len(churn_data), " customers in the data set and there are ", len(col_names), "features.")
#churn_data.columns() This is an index object, not a list, so can't just print it



#Surname
#np.unique(churn_data.Surname)
#len(np.unique(churn_data.Surname)) #2932 surprisingly low!
print("There are ", len(np.unique(churn_data.Surname)), "unique last names in the dataset." )

#Surnames could be an indicator of families. However, there are too many different surnames, some of which are common and with over 32 customers with the same last name.
churn_rate_by_surname = []
freq_surname = []
for i in np.unique(churn_data.Surname):
    freq_surname.append(sum(churn_data.Surname == i))
print("There are between ", min(freq_surname), " and ", max(freq_surname), " customers of each last name. This does not really allow us to deduce the existence of something like families from the surnames.")



#Credit score
#Je préfère le seaborn je pense
#plt.hist(churn_data.CreditScore)
print("The average credit score is ", np.mean(churn_data.CreditScore))
#np.median(churn_data.CreditScore)
sns.kdeplot(churn_data.CreditScore)
#The distribution is normal





#Geography

countries = np.unique(churn_data.Geography)
freq_country = [sum(churn_data.Geography == i) for i in countries]
#Better graph?
plt.bar(countries,freq_country)
for i in range(len(countries)):
    print("There are ",freq_country[i], "customers in "+countries[i])


#Gender
gender = np.unique(churn_data.Gender)
freq_gender = [sum(churn_data.Gender == i ) for i in gender]
plt.pie(freq_gender, labels = gender, autopct='%1.1f%%')

print(round(freq_gender[0]/100,1), "% of the customers are female. The rest are male.")


#Age
#plt.hist(churn_data.Age)
sns.kdeplot(churn_data.Age).set(title = "Distribution of customers' age")
#np.mean(churn_data.Age)
#np.median(churn_data.Age)
print("The average age of a customer in the data set is ",round(np.mean(churn_data.Age)), " years old")

#Tenure
freq_tenure = [sum(churn_data.Tenure == i ) for i in range(11)]
plt.bar(np.unique(churn_data.Tenure),freq_tenure) 
#sns.kdeplot(churn_data.Tenure) Not as relevant here because nb of years is discrete
print("The average customer has been at this bank for ", np.mean(churn_data.Tenure) )
#Important to note that there is no cutoff


#Balance
#plt.hist(churn_data.Balance) #A: Large number with very low
sns.kdeplot(churn_data.Balance).set(title = "Distribution of bank balance of customers")
#Seems to have a normal distribution, with the exception of the low balance?
print("The median bank balance is", np.median(churn_data.Balance))
#Median because of the bimodal distribution

#NumOfProducts
freq_NumOfProducts = [sum(churn_data.NumOfProducts == i ) for i in range(1,5)]
#plt.bar(range(1,5), freq_NumOfProducts) #Make this 1, 2, or 3 or 4, not 0.4
plt.pie(freq_NumOfProducts, labels = range(1,5), autopct='%1.1f%%') #Maybe not ideal...
print(round(freq_NumOfProducts[0]/100), "% of customers have one bank product. ",round(freq_NumOfProducts[1]/100), "% of customers have 2 products.")


#HasCrCard
label_HasCrCard = ['Has a credit card', 'Does not have a credit card']
freq_HasCrCard= [sum(churn_data.HasCrCard), 10000-sum(churn_data.HasCrCard) ] #A: OPTIMIZE!!!!!
plt.pie(freq_HasCrCard, labels = label_HasCrCard, autopct='%1.1f%%')
print(round(freq_HasCrCard[0]/100), "% of customers have a credit card. ",round(freq_HasCrCard[1]/100), "% of customers do not have a credit card.")



#IsActiveMember
label_IsActiveMember = ['Active member', 'Inactive member']
freq_IsActiveMember= [sum(churn_data.IsActiveMember), 10000-sum(churn_data.IsActiveMember) ] #A: OPTIMIZE!!!!!
plt.pie(freq_IsActiveMember, labels = label_IsActiveMember, autopct='%1.1f%%')
plt.title("Proportion of bank customers that are active")
print(round(freq_IsActiveMember[0]/100), "% of customers are active. ",round(freq_IsActiveMember[1]/100), "% of customers are not active.")


#EstimatedSalary
#plt.hist(churn_data.EstimatedSalary) #A: very ugly, find smth better
sns.kdeplot(churn_data.EstimatedSalary).set(title = "Distribution of customers' estimated salary") #Relatively homogenous
print("The average customer's estimated yearly salary is "+('{:,}'.format(round(np.mean(churn_data.EstimatedSalary)))) + " euros.") #Put less decimal points


##### What is the churn rate for the bank customers?
#/!\ For statistical significance, see below Eddy's stat tests

#the annual percentage rate at which customers leave
churn_rate = sum(churn_data.Exited)/len(churn_data)
print("The churn rate for all of the customers is ", round(churn_rate*100), "%")

churn_rate_genderF = sum(churn_data.Exited[churn_data.Gender == "Female"])/len(churn_data[churn_data.Gender == "Female"])
churn_rate_genderM = sum(churn_data.Exited[churn_data.Gender == "Male"])/len(churn_data[churn_data.Gender == "Male"])
print("The churn rate for women is ", round(churn_rate_genderF*100), "%")
print("The churn rate for men is ", round(churn_rate_genderM*100), "%")

for i in countries:
    churn_rate_country = sum(churn_data.Exited[churn_data.Geography == i])/len(churn_data[churn_data.Geography == i])
    print("The churn rate for "+i+" is ", round(churn_rate_country*100), "%")


#Now we drop it?
#/!\ Why?
#Att: change the name!!!!
churn_data = churn_data.drop(["CustomerId", "Surname"], axis = 1) 
#Att: need to get rid of Surname for the probit, so I kept it here
#Should probs give it another name

##### What is the relationships between variables ? 
#Replacing gender by numerical values 
churn_data["Gender"] = churn_data["Gender"].replace({"Female": 1, "Male": 0})

#simple covariance matrix
correlation_matrix = churn_data.corr()
print(correlation_matrix) 
#In general what we can remark is that explicative variables are not that much correlated with one another, 
#which is good thing since it implies more explicative power for the churn rate.


#Pretty covariance matrix, maybe see about the colors

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})



### For the age variable 

#Activity 
sns.lineplot(x="Age", y="IsActiveMember", data=churn_data, marker="o")
#I like th


##For all other variables, the mean stays constant, for Balance and Estimated Salary it is 
###What is the distribution of variables across countries ?
#Clients count
country_count = churn_data.groupby("Geography").size().reset_index(name="Count")
custom_colors = ["blue", "green", "orange"]

#Att: ne marche pas, grouped_data n'a pas d'attribut Geography et n'est défini qu'après
#plt.bar(grouped_data['Geography'], grouped_data['Count'], color = custom_colors)
#plt.ylabel('Number of clients')


#Suggestion An: est-ce que c'est ce que tu voulais? On l'avait déjà au dessus je crois
plt.bar(country_count["Geography"], country_count["Count"], color = custom_colors)
plt.ylabel("Number of clients")

#Distributions
grouped_data = churn_data.groupby("Geography").mean().reset_index()

sns.boxplot(y="CreditScore", x="Geography", hue="Geography", data=churn_data).set(xlabel='')
plt.gca().legend().remove()

sns.boxplot(y="Tenure",x = "Geography", hue = "Geography", data = churn_data).set(xlabel='')
plt.gca().legend().remove()

sns.boxplot(y="Balance",x = "Geography", hue = "Geography",data = churn_data).set(xlabel='')
plt.gca().legend().remove()

sns.boxplot(y="EstimatedSalary",x = "Geography", hue = "Geography",data = churn_data).set(xlabel='')
plt.gca().legend().remove()

#Att question An: what are we trying to show with these boxplots? Is there a more illustrative way of showing smth?




### Influence of other variables on churn --> je ne comprends pas ce que tu veux faire?
sns.countplot(data=churn_data, x="Exited", hue="Gender") #Refaire ça avec les variables au début (avec female et male plutôt)
sns.countplot(data=churn_data, x="Exited", hue="Geography")

#Higher churn rate for female compared to male
#Higher churn rate in Germany and spain compared to france, maybe a link with the number of clients

#/!\ An: tu ne calcules pas le churn rate, juste le nb de exits?
#Je réflechis à qqch de plus parlant?

#Are they significative ? 
from scipy.stats import ttest_ind

#Separate the churn variable according to gender and geography
female_data = churn_data[churn_data["Gender"] == 1]["Exited"]
male_data = churn_data[churn_data["Gender"] == 0]["Exited"]

france_data= churn_data[churn_data["Geography"] == "France"]["Exited"]
spain_data= churn_data[churn_data["Geography"] == "Spain"]["Exited"]
germany_data= churn_data[churn_data["Geography"] == "Germany"]["Exited"]

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

#We run a probit regression to look at:
    #1) which variables have a statistically significant impact on churn
    #2) which direction (postive/negative) the variables impact churn
    
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.tools import add_constant

churn_data["Geography"] = churn_data["Geography"].replace({"France": 1, "Spain": 2, "Germany":3})


#print(churn_data.head())
#churn_data.describe()

Y = churn_data["Exited"]
X = churn_data.drop(["Exited"], 1)
X = add_constant(X) #Att why this?
model = Probit(Y, X.astype(float))
probit_model = model.fit()
print(probit_model.summary())






### Machine learning Model ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Without the country variable : 
#An: Why?

Y = churn_data["Exited"]
X = churn_data.drop("Exited", axis = 1)
X = X.drop("Geography", axis = 1)

#Spliting the sample
#J'ajoute stratify
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42, stratify=Y)

#looping for best values of min_samples_split 

accuracy_values= []

for i in range(2, 50): 
    model = RandomForestClassifier(min_samples_split=i, random_state = 42)
    model.fit(X_train, Y_train)
    prediction_Y = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction_Y)
    accuracy_values.append(accuracy)



#With the country variable, using one-hot encoding 
#Creating a new dataset to not conflict with the previous one

df = pd.read_csv("churn_bank.txt", sep = ",")
df["Gender"] = df["Gender"].replace({"Female": 1, "Male": 0})
df = df.drop(["CustomerId", "Surname"], axis = 1)

#Adding dummies for the presence or not of one country in the row 

df = pd.get_dummies(df, columns=["Geography"], prefix='geo', drop_first=False)

Y = df["Exited"]
X = df.drop("Exited", axis = 1)

#splitting the sample

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42, stratify = Y)

#looping for best min_samples_split
accuracy_values = []

for i in range(2, 50): 
    model = RandomForestClassifier(min_samples_split=i, random_state = 42)
    model.fit(X_train, Y_train)
    prediction_Y = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction_Y)
    accuracy_values.append(accuracy)

#accury over different values of min_samples_split 

min_samples_split=list(range(2,50))
plt.plot(min_samples_split, accuracy_values, marker='o')
plt.title("Change in mse with different min samples split values")
plt.xlabel("min_sample_split")
plt.ylabel("accuracy")
plt.grid(True)

#looping for best n_estimator
accuracy_values = []

for i in range(80, 200): 
    model = RandomForestClassifier(n_estimators = i, min_samples_split=27, random_state = 42)
    model.fit(X_train, Y_train)
    prediction_Y = model.predict(X_test)
    accuracy = accuracy_score(Y_test, prediction_Y)
    accuracy_values.append(accuracy)

#accuracy across n_estimator with best min_samples_split

n_regressors=list(range(80,200))
plt.plot(n_regressors, accuracy_values, marker='o')
plt.title("Change in mse with different min samples split values")
plt.xlabel("n_regressors")
plt.ylabel("accuracy")
plt.grid(True)


#1) Best value for min_samples_split is 42, Accuracy = 0.86
#2) Best value for min_samples_split is 27, best value for n_estimators is 120, Accuracy = 0.875
#Ici je suis très con parce que je me suis cassé la tete à faire 4 loops alors que ca se résume à deux grid search, il faut que je modifie ca

#With the country variable as a numerical label 





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
