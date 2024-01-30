# Titanic - Machine Learning from Disaster 
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

import seaborn as sns
from collections import Counter
```
## Introduction
The sinking of Titanic is one of the most notorious shipwrecks in the history. In 1912, during her voyage, the titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. <br>
Content:
1. [Load and Check Data](#1)
2. [Variable Description](#2)
   * [Univariate Variable Analysis](#3)
      * [Categorical Variable Analysis](#4)
      * [Numerical Variable Analysis](#5)
3. [Basic Data Analysis](#6)
4. [Outlier Detection](#7)
5. [Missing Value](#8)
   * [Find Missing Value](#9)
   * [Fill Missing Value](#10)

### Load and Check Data 
```python
train_pdf = pd.read_csv('/kaggle/input/titanic/train.csv')
test_pdf = pd.read_csv('/kaggle/input/titanic/train.csv')
test_PassengerId = test_df["PassengerId"]
```
```python
train_df.columns
```
Index (['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'], dtype = 'object')

```python
train_df.head()
``` 
| PassengerId | Survived | Pclass | Name                                              | Sex    | Age  | SibSp | Parch | Ticket         | Fare     | Cabin | Embarked |
|--------------|----------|--------|---------------------------------------------------|--------|------|-------|-------|----------------|----------|-------|----------|
| 1            | 0        | 3      | Braund, Mr. Owen Harris                           | male   | 22.0 | 1     | 0     | A/5 21171      | 7.2500   | NaN   | S        |
| 2            | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer)| female | 38.0 | 1     | 0     | PC 17599       | 71.2833  | C85   | C        |
| 3            | 1        | 3      | Heikkinen, Miss. Laina                             | female | 26.0 | 0     | 0     | STON/O2. 3101282| 7.9250   | NaN   | S        |
| 4            | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel)       | female | 35.0 | 1     | 0     | 113803         | 53.1000  | C123  | S        |
| 5            | 0        | 3      | Allen, Mr. William Henry                            | male   | 35.0 | 0     | 0     | 373450         | 8.0500   | NaN   | S        |
```python
train_df.describe()
```
|           | PassengerId | Survived | Pclass | Age       | SibSp | Parch | Fare       |
|-----------|-------------|----------|--------|-----------|-------|-------|------------|
| **count** | 891.000000  | 891.0000 | 891.00 | 714.000000| 891.0 | 891.0 | 891.000000 |
| **mean**  | 446.000000  | 0.383838 | 2.3086 | 29.699118 | 0.523 | 0.382 | 32.204208  |
| **std**   | 257.353842  | 0.486592 | 0.8361 | 14.526497 | 1.103 | 0.806 | 49.693429  |
| **min**   | 1.000000    | 0.000000 | 1.0000 | 0.420000  | 0.000 | 0.000 | 0.000000   |
| **25%**   | 223.500000  | 0.000000 | 2.0000 | 20.125000 | 0.000 | 0.000 | 7.910400   |
| **50%**   | 446.000000  | 0.000000 | 3.0000 | 28.000000 | 0.000 | 0.000 | 14.454200  |
| **75%**   | 668.500000  | 1.000000 | 3.0000 | 38.000000 | 1.000 | 0.000 | 31.000000  |
| **max**   | 891.000000  | 1.000000 | 3.0000 | 80.000000 | 8.000 | 6.000 | 512.329200 |

### Variable Description
1. PassengerId: unique id number to each passenger
2. Survived: passenger survive (1) or died (0)
3. Pclass: passenger class
4. Name: name
5. Sex: gender of passenger
6. Age: age of passenger
7. SibSip: number of siblings / spouses
8. Parch: number of parents / children
9. Ticket: ticket number
10. Fare: amount of money spent on ticket
11. Cablin: cabin category
12. Embarked: port where passenger embarked (C=Cherboug, Q=Queenstown, S=Southampton)
```python
train_df.info()
```
| Column       | Non-Null Count | Dtype    |
|--------------|----------------|----------|
| PassengerId  | 891            | int64    |
| Survived     | 891            | int64    |
| Pclass       | 891            | int64    |
| Name         | 891            | object   |
| Sex          | 891            | object   |
| Age          | 714            | float64  |
| SibSp        | 891            | int64    |
| Parch        | 891            | int64    |
| Ticket       | 891            | object   |
| Fare         | 891            | float64  |
| Cabin        | 204            | object   |
| Embarked     | 889            | object   |


float64(2): Fare and Age <br>
int64(5): PassengerId, Survived, Pclass, SibSp, Parch <br>
object(5): Name, Sex, Ticket, Cabin, Embarked

#### Univariate Variable Analysis 
* Categorical Variable: Survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp and Parch
* Numerical Variable: Fare, Age and PassengerId

##### Categorical Variable 
```python
def bar_plot(variable):
  var = train_df[variable]
  varValue = var.value_counts()

  plt.figure(figsize = (9,3))
  plt.bar(varValue.index, varValue)
  plt.xticks(varValue.index, varValue.index.values)
  plt.ylabel("Frequency")
  plt.title(variable)
  plt.show()
  print("{}: \n {}".format(variable, varValue))
```
```python
category1 = ["Survived", "Sex", "Pclass", "Embarked", "SibSp", "Parch"]
for c in category1:
  bar_plot(c)
```
![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/0e96402b-9b1a-4c28-872d-7e31a327228c)

|   | Survived | count |
|---|----------|-------|
| 0 | 0        | 549   |
| 1 | 1        | 342   |

![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/72d9d4bf-dc66-4403-8b15-59d941d6f7d3)

|       | Sex    | count |
|-------|--------|-------|
| male  | male   | 577   |
| female| female | 314   |

![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/ed6559c6-2731-43ad-8398-447816662300)

|   | Pclass | count |
|---|--------|-------|
| 3 | 3      | 491   |
| 1 | 1      | 216   |
| 2 | 2      | 184   |

![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/1b70aa80-b449-4820-be1e-a0a043ff86b8)

|   | Embarked | count |
|---|----------|-------|
| 0 | S        | 644   |
| 1 | C        | 168   |
| 2 | Q        | 77    |

![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/af66f7ee-de6a-46b4-9a3a-f70fc3420d78)

|   | SibSp | count |
|---|-------|-------|
| 0 | 0     | 608   |
| 1 | 1     | 209   |
| 2 | 2     | 28    |
| 3 | 3     | 16    |
| 4 | 4     | 18    |
| 5 | 5     | 5     |
| 8 | 8     | 7     |

![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/7dc5634e-12f3-489b-9c8b-f3925253f009)

|   | Parch | count |
|---|-------|-------|
| 0 | 0     | 678   |
| 1 | 1     | 118   |
| 2 | 2     | 80    |
| 3 | 3     | 5     |
| 4 | 4     | 4     |
| 5 | 5     | 5     |
| 6 | 6     | 1     |

```python
category2 = ["Cabin", "Name", "Ticket"]
for c in category2:
  print("{} \n".format(train_df[c].value_counts))
```
|   | Cabin | count |
|---|-------|-------|
| 0 | NaN   | 687   |
| 1 | C85   | 1     |
| 2 | C123  | 1     |
| 3 | C148  | 1     |
| 4 | B96   | 1     |
|...| ...   | ...   |
| 886 | NaN | 687   |
| 887 | B42 | 1     |
| 888 | NaN | 687   |
| 889 | C148| 1     |
| 890 | NaN | 687   |

|   | Name                                                | count |
|---|-----------------------------------------------------|-------|
| 0 | Braund, Mr. Owen Harris                             | 1     |
| 1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | 1     |
| 2 | Heikkinen, Miss. Laina                               | 1     |
| 3 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | 1     |
| 4 | Allen, Mr. William Henry                             | 1     |
|...| ...                                                 | ...   |
| 886 | Montvila, Rev. Juozas                              | 1     |
| 887 | Graham, Miss. Margaret Edith                       | 1     |
| 888 | Johnston, Miss. Catherine Helen "Carrie"           | 1     |
| 889 | Behr, Mr. Karl Howell                               | 1     |
| 890 | Dooley, Mr. Patrick                                 | 1     |

|   | Ticket         | count |
|---|----------------|-------|
| 0 | A/5 21171      | 1     |
| 1 | PC 17599       | 1     |
| 2 | STON/O2. 3101282| 1     |
| 3 | 113803         | 1     |
| 4 | 373450         | 1     |
|...| ...            | ...   |
| 886 | 211536       | 1     |
| 887 | 112053       | 1     |
| 888 | W./C. 6607    | 1     |
| 889 | 111369       | 1     |
| 890 | 370376       | 1     |

##### Numerical Variable 
```python
def plot_hist(variable):
    plt.figure(figsize = (9,2))
    plt.hist(train_df[variable], bins=50)
    plt.xlabel(variable)
    plt.ylabel("Frequency")
    plt.title("{} distribution with hist".format(variable))
    plt.show()
```
```python
numericVar = ["Fare","Age","PassengerId"]
for n in numericVar:
    plot_hist(n)
```
![image](https://github.com/beyzaturku/Kaggle_Titanic/assets/75912974/3274e50f-513f-407a-bb9e-725b435d0d08)

### Basic Data Analysis
* Pclass - Survived
* Sex - Survived (What is the relationship between gender and survival?)
* SibSp - Survived
* Parch - Survived
```python
#Pclass and Survived
train_df[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean().sort_values(by="Survived", ascending=False)
#Group the pclass and survived data take the average.
```
|   | Pclass | Survived |
|---|--------|----------|
| 0 | 1      | 0.629630 |
| 1 | 2      | 0.472826 |
| 2 | 3      | 0.242363 |

```python
#Sex and Survived
train_df[["Sex","Survived"]].groupby(["Sex"],as_index = False).mean()
```
|   | Sex    | Survived |
|---|--------|----------|
| 0 | female | 0.742038 |
| 1 | male   | 0.188908 |

```python
#SibSp and Survived
train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index = False).mean().sort_values(by="Survived",ascending=False)
```
|   | SibSp | Survived |
|---|-------|----------|
| 1 | 1     | 0.535885 |
| 2 | 2     | 0.464286 |
| 0 | 0     | 0.345395 |
| 3 | 3     | 0.250000 |
| 4 | 4     | 0.166667 |
| 5 | 5     | 0.000000 |
| 6 | 8     | 0.000000 |

### Outlier Detection
```python
def detect_outliers(df, features):
    outlier_indices = []
    
    for c in features:
        #1st quartile
        Q1 = np.percentile(df[c],25)
        #3rd quartile
        Q3 = np.percentile(df[c],75)
        #IQR
        IQR = Q3 - Q1
        #Outlier step 
        outlier_step = IQR * 1.5 
        #detect outlier and their indeces 
        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 - outlier_step)].index
        # store indeces
        outlier_indices.extend(outlier_list_col)
        
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 2)
    
    return multiple_outliers
```
```python
train_df.loc[detect_outliers(train_df,["Age","SibSp","Parch","Fare"])]
```
|   | PassengerId | Survived | Pclass | Name                                                 | Sex    | Age  | SibSp | Parch | Ticket    | Fare     | Cabin | Embarked |
|---|--------------|----------|--------|------------------------------------------------------|--------|------|-------|-------|-----------|----------|-------|----------|
| 7 | 8            | 0        | 3      | Palsson, Master. Gosta Leonard                      | male   | 2.0  | 3     | 1     | 349909    | 21.0750  | NaN   | S        |
| 8 | 9            | 1        | 3      | Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)   | female | 27.0 | 0     | 2     | 347742    | 11.1333  | NaN   | S        |
|10 | 11           | 1        | 3      | Sandstrom, Miss. Marguerite Rut                      | female | 4.0  | 1     | 1     | PP 9549   | 16.7000  | G6    | S        |
|13 | 14           | 0        | 3      | Andersson, Mr. Anders Johan                          | male   | 39.0 | 1     | 5     | 347082    | 31.2750  | NaN   | S        |
|16 | 17           | 0        | 3      | Rice, Master. Eugene                                 | male   | 2.0  | 4     | 1     | 382652    | 29.1250  | NaN   | Q        |
|...| ...          | ...      | ...    | ...                                                  | ...    | ...  | ...   | ...   | ...       | ...      | ...   | ...      |
|871 | 872          | 1        | 1      | Beckwith, Mrs. Richard Leonard (Sallie Monypeny)    | female | 47.0 | 1     | 1     | 11751     | 52.5542  | D35   | S        |
|879 | 880          | 1        | 1      | Potter, Mrs. Thomas Jr (Lily Alexenia Wilson)       | female | 56.0 | 0     | 1     | 11767     | 83.1583  | C50   | C        |
|880 | 881          | 1        | 2      | Shelley, Mrs. William (Imanita Parrish Hall)        | female | 25.0 | 0     | 1     | 230433    | 26.0000  | NaN   | S        |
|885 | 886          | 0        | 3      | Rice, Mrs. William (Margaret Norton)                | female | 39.0 | 0     | 5     | 382652    | 29.1250  | NaN   | Q        |
|888 | 889          | 0        | 3      | Johnston, Miss. Catherine Helen "Carrie"           | female | NaN  | 1     | 2     | W./C. 6607| 23.4500  | NaN   | ...      |

```python
train_df = train_df.drop(detect_outliers(train_df, ["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)
```
### Missing Value 
* Find Missing Value
* Fill Missing Value
```python
train_df_len = len(train_df)
traind_df = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)
```
```python
train_df.head()
```
|   | PassengerId | Survived | Pclass | Name                                               | Sex    | Age  | SibSp | Parch | Ticket         | Fare     | Cabin | Embarked |
|---|--------------|----------|--------|----------------------------------------------------|--------|------|-------|-------|----------------|----------|-------|----------|
| 0 | 1            | 0        | 3      | Braund, Mr. Owen Harris                            | male   | 22.0 | 1     | 0     | A/5 21171      | 7.2500   | NaN   | S        |
| 1 | 2            | 1        | 1      | Cumings, Mrs. John Bradley (Florence Briggs Thayer)| female | 38.0 | 1     | 0     | PC 17599       | 71.2833  | C85   | C        |
| 2 | 3            | 1        | 3      | Heikkinen, Miss. Laina                              | female | 26.0 | 0     | 0     | STON/O2. 3101282| 7.9250   | NaN   | S        |
| 3 | 4            | 1        | 1      | Futrelle, Mrs. Jacques Heath (Lily May Peel)       | female | 35.0 | 1     | 0     | 113803         | 53.1000  | C123  | S        |
| 4 | 5            | 0        | 3      | Allen, Mr. William Henry                            | male   | 35.0 | 0     | 0     | 373450         | 8.0500   | NaN   | S        |

#### Find Missing Value 
```python
train_df.columns[train_df.isnull().any()]
```
Index(['Age', 'Cabin', 'Embarked'], dtype='object')
```python
train_df.isnull().sum()
```
|             | Count |
|-------------|-------|
| PassengerId | 0     |
| Survived    | 0     |
| Pclass      | 0     |
| Name        | 0     |
| Sex         | 0     |
| Age         | 157   |
| SibSp       | 0     |
| Parch       | 0     |
| Ticket      | 0     |
| Fare        | 0     |
| Cabin       | 537   |
| Embarked    | 2     |

#### Fill Missing Value 
* Embarked has 2 missing value
* Fare has only 1
```python
train_df[train_df["Embarked"].isnull()]
```
|   | PassengerId | Survived | Pclass | Name                                       | Sex    | Age  | SibSp | Parch | Ticket | Fare | Cabin | Embarked |
|---|--------------|----------|--------|--------------------------------------------|--------|------|-------|-------|--------|------|-------|----------|
| 48| 62           | 1        | 1      | Icard, Miss. Amelie                        | female | 38.0 | 0     | 0     | 113572 | 80.0 | B28   | NaN      |
|633| 830          | 1        | 1      | Stone, Mrs. George Nelson (Martha Evelyn) | female | 62.0 | 0     | 0     | 113572 | 80.0 | B28   | NaN      |












