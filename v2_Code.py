# -*- coding: utf-8 -*-
"""
Python project due November 17th 2023

@author: Anastasia and Eddy
"""

import pandas as pd 
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
churn_data =  pd.read_csv("churn_bank.txt", delimiter=",")
#churn_data = churn_data.drop(["CustomerId", "Surname"], axis = 1) 
#Attention: pourquoi est-ce qu'on drop CustomerId et Surname? Nécessaire pourtant

churn_data.isnull().sum()
#No missing values



##### Getting to know the dataset

##Setting parameters for graphical representations
#Setting our favorite colours: 
colors = ["#FFD699", "#87CEEB"]
#To separate the components of our pie charts 
explode = (0.1, 0)  

col_names = churn_data.columns.values.tolist()
print("There are ",len(churn_data), " customers in the data set and there are ", len(col_names), "features.")

#Surname
print("There are ", len(np.unique(churn_data.Surname)), "unique last names in the dataset." )

#Surnames could be an indicator of families. However, there are too many different surnames, some of which are common and with over 32 customers with the same last name.
churn_rate_by_surname = []
freq_surname = []
for i in np.unique(churn_data.Surname):
    freq_surname.append(sum(churn_data.Surname == i))
print("There are between ", min(freq_surname), " and ", max(freq_surname), " customers of each last name. This does not really allow us to deduce the existence of something like families from the surnames.")


#Credit score
print("The average credit score is ", np.mean(churn_data.CreditScore))
plt.figure(figsize=(5.5, 3.5))
sns.kdeplot(churn_data["CreditScore"], fill=True, color="skyblue", linewidth=2)
plt.show()


#Geography
countries = np.unique(churn_data.Geography)
freq_country = [sum(churn_data.Geography == i) for i in countries]
sns.barplot(x= countries, y= freq_country, color=colors[1])
plt.show()
for i in range(len(countries)):
    print("There are ",freq_country[i], "customers in "+countries[i])


#Gender
gender = np.unique(churn_data.Gender)
freq_gender = [sum(churn_data.Gender == i ) for i in gender]

plt.pie(freq_gender, labels = gender, autopct="%1.1f%%", startangle =90, colors=colors, explode = explode, shadow=True)
plt.show()
print(round(freq_gender[0]/100,1), "% of the customers are female. The rest are male.")


#Age
plt.figure(figsize=(5.5, 3.5))
sns.kdeplot(churn_data["Age"], fill=True, color="skyblue", linewidth=2)
plt.show()

print("The average age of a customer in the data set is ",round(np.mean(churn_data.Age)), " years old")


#Tenure
freq_tenure = [sum(churn_data.Tenure == i ) for i in range(11)]
sns.barplot(x = np.unique(churn_data.Tenure),y = freq_tenure, color=colors[1]).set(title = "Distribution of customers by number of years of tenure")
plt.show()
print("The average customer has been at this bank for ", np.mean(churn_data.Tenure) )


#Balance
sns.kdeplot(churn_data.Balance, fill=True, color="skyblue", linewidth=2)
print("The median bank balance is", np.median(churn_data.Balance))
#Median because of the bimodal distribution


#NumOfProducts
freq_NumOfProducts = [sum(churn_data.NumOfProducts == i ) for i in range(1,5)]
plt.pie(freq_NumOfProducts, labels = range(1,5), startangle =90, colors=colors, explode = (0.1, 0,0,0), shadow=True)
plt.show()
print(round(freq_NumOfProducts[0]/100), "% of customers have one bank product. ",round(freq_NumOfProducts[1]/100), "% of customers have 2 products.")


#HasCrCard
label_HasCrCard = ["Has a credit card", "Does not have a credit card"]
freq_HasCrCard= [sum(churn_data.HasCrCard), 10000-sum(churn_data.HasCrCard) ] 
plt.pie(freq_HasCrCard, labels = label_HasCrCard, autopct="%1.1f%%", startangle = 90,  colors=colors, explode = explode, shadow=True)
plt.show()
print(round(freq_HasCrCard[0]/100), "% of customers have a credit card. ",round(freq_HasCrCard[1]/100), "% of customers do not have a credit card.")


#IsActiveMember
label_IsActiveMember = ["Active member", "Inactive member"]
freq_IsActiveMember= [sum(churn_data.IsActiveMember), 10000-sum(churn_data.IsActiveMember) ]

plt.pie(freq_IsActiveMember, labels = label_IsActiveMember, autopct="%1.1f%%", startangle = 90,  colors=colors, explode = explode, shadow=True)
plt.title("Proportion of bank customers that are active")
plt.show()
print(round(freq_IsActiveMember[0]/100), "% of customers are active. ",round(freq_IsActiveMember[1]/100), "% of customers are not active.")


#EstimatedSalary
sns.kdeplot(churn_data.EstimatedSalary, fill=True, color="skyblue", linewidth=2).set(title = "Distribution of customers' estimated salary")
plt.show()
print("The average customer's estimated yearly salary is "+("{:,}".format(round(np.mean(churn_data.EstimatedSalary)))) + " euros.") 


##### What is the churn rate for the bank customers?

#The annual percentage rate at which customers leave
churn_rate = sum(churn_data.Exited)/len(churn_data)
print("The churn rate for all of the customers is ", round(churn_rate*100), "%")

sns.countplot(data=churn_data, x="Exited", hue="Gender", palette = colors*2) 
plt.show()

churn_rate_genderF = sum(churn_data.Exited[churn_data.Gender == "Female"])/len(churn_data[churn_data.Gender == "Female"])
churn_rate_genderM = sum(churn_data.Exited[churn_data.Gender == "Male"])/len(churn_data[churn_data.Gender == "Male"])
print("The churn rate for women is ", round(churn_rate_genderF*100), "%")
print("The churn rate for men is ", round(churn_rate_genderM*100), "%")

for i in countries:
    churn_rate_country = sum(churn_data.Exited[churn_data.Geography == i])/len(churn_data[churn_data.Geography == i])
    print("The churn rate for "+i+" is ", round(churn_rate_country*100), "%")


#Dropping the two non-explicative variables

churn_data_v2 = churn_data.drop(["CustomerId", "Surname"], axis = 1) 


##### What is the relationships between variables ? 
#Replacing gender by numerical values 
churn_data_v2["Gender"] = churn_data_v2["Gender"].replace({"Female": 1, "Male": 0})

#simple covariance matrix
correlation_matrix = churn_data_v2.corr()
print(correlation_matrix) 

#Illustrative covariance matrix
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap_custom = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap_custom, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


### For the age variable 
 
plt.xlim(16, 80)
sns.lineplot(x="Age", y="IsActiveMember", data= churn_data_v2, marker="o", color="blue")
plt.ylabel("Client activity")
plt.show()

plt.xlim(16, 80)
sns.lineplot(x="Age", y="Exited", data=churn_data_v2, marker="o", color="blue")
plt.ylabel("Churn Rate")
plt.show()


##Geography

#Balance distribution across countries
sns.boxplot(y="Balance",x = "Geography", hue = "Geography",data = churn_data_v2).set(xlabel="")
plt.gca().legend().remove()
plt.show()

sns.countplot(data=churn_data_v2, x="Exited", hue="Geography")
plt.show()


### Influence of other variables on churn 
sns.countplot(data=churn_data_v2, x="Exited", hue="Gender") 


#Are the differences statistically significant ? 
from scipy.stats import ttest_ind

#Separate the churn variable according to gender and geography
female_data = churn_data_v2[churn_data_v2["Gender"] == 1]["Exited"]
male_data = churn_data_v2[churn_data_v2["Gender"] == 0]["Exited"]

france_data= churn_data_v2[churn_data_v2["Geography"] == "France"]["Exited"]
spain_data= churn_data_v2[churn_data_v2["Geography"] == "Spain"]["Exited"]
germany_data= churn_data_v2[churn_data_v2["Geography"] == "Germany"]["Exited"]

#For gender
t_statistic, p_value = ttest_ind(female_data, male_data)
print("The difference in churn rate between men and women is statistically significant: ",  f"P-value is {p_value} and ", f"T-statistic: {t_statistic}")

#For geography 
t_statistic, p_value = ttest_ind(france_data, spain_data)
print("The difference in churn rate between France and Spain is not statistically significant: ",  f"P-value is {p_value} and ", f"T-statistic is {t_statistic}")
#Not significant

t_statistic, p_value = ttest_ind(france_data, germany_data)
print("The difference in churn rate between France and Germany is statistically significant: ",  f"P-value is {p_value} and ", f"T-statistic is {t_statistic}")

t_statistic, p_value = ttest_ind(spain_data, germany_data)
print("The difference in churn rate between Spain and Germany is statistically significant: ",  f"P-value is {p_value} and ", f"T-statistic is {t_statistic}")


#We run a probit regression to look at:
    #1) which variables have a statistically significant impact on churn
    #2) which direction (postive/negative) the variables impact churn
    
from statsmodels.discrete.discrete_model import Probit
from statsmodels.tools.tools import add_constant

churn_data_v2["Geography"] = churn_data_v2["Geography"].replace({"France": 1, "Spain": 2, "Germany":3})

Y = churn_data_v2["Exited"]
X = churn_data_v2.drop(labels= ["Exited"], axis=1)
X = add_constant(X) 
model = Probit(Y, X.astype(float))
probit_model = model.fit()
print(probit_model.summary())







### Machine learning Model ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#With the country variable, using one-hot encoding 
#Creating a new dataset to not conflict with the previous one

churn_data_ml = pd.read_csv("churn_bank.txt", sep = ",")
churn_data_ml["Gender"] = churn_data_ml["Gender"].replace({"Female": 1, "Male": 0})
churn_data_ml = churn_data_ml.drop(["CustomerId", "Surname"], axis = 1)

#Adding dummies for the presence or not of one country in the row 
churn_data_ml = pd.get_dummies(churn_data_ml, columns=["Geography"], prefix="geo", drop_first=False)

Y = churn_data_ml["Exited"]
X = churn_data_ml.drop("Exited", axis = 1)

#splitting the sample
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42, stratify = Y)

#Grid search to find best parameters for RF: n_estimators (the number of trees) and min_samples_split (The minimum number of samples required to split an internal node) 

churn_predict_model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": list(range(95, 100)),
    "min_samples_split": list(range(10, 20))}

grid_search = GridSearchCV(churn_predict_model, param_grid=param_grid, cv=3, n_jobs=12)
grid_search.fit(X_train, Y_train)    
best_model = grid_search.best_estimator_
Y_predicted = best_model.predict(X_test)
accuracy_best = accuracy_score(Y_test, Y_predicted)

print("The best parameters:", grid_search.best_params_)
print(f"Accuracy (Best Model): {accuracy_best:.4f}")
