import pandas as pd

df = pd.read_csv("Titanic-Dataset - Titanic-Dataset.csv")

df.head()

df.isnull().sum()

df.corr(numeric_only=True)

df = df.drop('PassengerId', axis=1)

df.shape

df.isnull().sum()

df.info()

df.head()

df.tail()

y = df['Survived']

y.shape

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

df = pd.get_dummies(df, columns=['Gender', 'Embarked'], drop_first=True)

X = df[['Pclass', 'Gender_male', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.shape

X_test.shape

y_train.shape

df.info()

df['Gender_male']

df.info()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

cm = confusion_matrix(y_test, y_pred)

print(cm)