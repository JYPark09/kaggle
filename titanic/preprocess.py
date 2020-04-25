import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

age_mean = np.mean(train.Age)
fare_mean = np.mean(train.Fare)

# Set for missing data
def fill_missing(df):
    df.Age = df.Age.fillna(age_mean)
    df.Embarked = df.Embarked.fillna('S') # S is most
    df.Fare = df.Fare.fillna(fare_mean)

    df['family'] = df.SibSp + df.Parch + 1

    df.Sex = df.Sex.replace('male', 0)
    df.Sex = df.Sex.replace('female', 1)

    df.Embarked = df.Embarked.replace('Q', 0)
    df.Embarked = df.Embarked.replace('C', 1)
    df.Embarked = df.Embarked.replace('S', 0)

fill_missing(train)
fill_missing(test)

print('[Train dataset info]')
train.info()
print()

print('[Test dataset info]')
test.info()
print()

train_np = train[['PassengerId', 'Pclass', 'Sex', 'Age', 'family', 'Fare', 'Embarked', 'Survived']].values
test_np = test[['PassengerId', 'Pclass', 'Sex', 'Age', 'family', 'Fare', 'Embarked']].values

np.save('data/train.npy', train_np)
np.save('data/test.npy', test_np)

