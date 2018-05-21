import pandas as pd
from pandas.plotting import scatter_matrix
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
sns.set_style("whitegrid")

# Read in data
titanic_df = pd.read_csv("C:\\Users\\tharu\\Dropbox\\PycharmProjects\\titanic_project\\data\\train.csv")
test_df = pd.read_csv("C:\\Users\\tharu\\Dropbox\\PycharmProjects\\titanic_project\\data\\test.csv")


# Write keys to text file for reference
text_file_path = Path('keys.txt')
if text_file_path.is_file() == False:
    with open('keys.txt', 'w') as f:
        f.write(str(titanic_df.keys()))

# Understand the data
print(titanic_df.head())
print(titanic_df.info())

# Feature engineering
titanic_df['Family_Size'] = titanic_df['SibSp'] + titanic_df['Parch']
titanic_df = titanic_df.drop(["SibSp", "Parch"], axis=1)


def passenger_type(parameters):
    age, sex = parameters
    return sex if age < 16 else "Child"

titanic_df["Person"] = titanic_df[["Age", "Sex"]].apply(passenger_type, axis=1)
titanic_df = titanic_df.drop(["Age", "Sex"], axis=1)
titanic_df = titanic_df.drop(["Ticket"], axis=1)  # Remove ticket as it doesn't provide useful information
titanic_df = titanic_df.drop(["Name"], axis=1)  # For same reason remove Name
titanic_df = titanic_df.drop(["PassengerId"], axis=1)

fare_survived = titanic_df[["Fare", "Survived"]].query("Survived == 1").drop(["Survived"], axis=1)
fare_not_survived = titanic_df[["Fare", "Survived"]].query("Survived == 0").drop(["Survived"], axis=1)

fare_survived_average = fare_survived.mean()
fare_survived_std = fare_survived.std()

fare_not_survived_average = fare_not_survived.mean()
fare_not_survived_std = fare_not_survived.std()

# Plot mean and std
average_fare = pd.DataFrame([fare_survived_average, fare_not_survived_average])
std_fare = pd.DataFrame([fare_survived_std, fare_not_survived_std])
average_fare_plot = average_fare.plot(yerr=std_fare, kind="bar")
average_fare_plot.set_xlabel("Survived")
average_fare_plot.set_ylabel("Average Fare")
plt.show()

# Check how import Embarked is for survival
titanic_df["Embarked"] = titanic_df['Embarked'].fillna("S")
sns.factorplot("Embarked", "Survived", data=titanic_df, kind="bar", ci=None)
plt.show()

# Create new category based on first letter of cabin, i.e. deck
titanic_df["Deck"] = titanic_df.Cabin.str[0]
titanic_df["Deck"] = titanic_df["Deck"].fillna("NaN")
titanic_df = titanic_df.drop(["Cabin"], axis=1)
sns.countplot("Deck", data=titanic_df)
plt.show()

# Create dummies for categorical data
dummies = pd.get_dummies(titanic_df["Person"])
dummies = dummies.drop(["male"], axis=1)

titanic_df = titanic_df.drop(["Person"], axis=1)
titanic_df = titanic_df.join(dummies)

titanic_df = titanic_df.drop(["Deck"], axis=1)

titanic_df = titanic_df.drop(["Embarked"], axis=1)

# Define train and test data
X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]

# Logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print(logreg.score(X_train, Y_train))

# Support vector machine
svc = SVC()
svc.fit(X_train, Y_train)
print(svc.score(X_train, Y_train))

# Random Forests
random_forest = RandomForestClassifier(n_estimators=100,oob_score=True,max_features=5)
random_forest.fit(X_train, Y_train)
print(random_forest.score(X_train, Y_train))

# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
print(coeff_df)













