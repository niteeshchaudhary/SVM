import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def calc_accu(x_train, y_train, x_test, y_test, classifier):
    classifier.fit(x_train, y_train)
    y_pred_train = classifier.predict(x_train)
    y_pred_test = classifier.predict(x_test)
    return accuracy_score(y_train, y_pred_train),accuracy_score(y_test, y_pred_test)

data = pd.read_csv('spambase.data', sep=',', header=None)
print(data.head())
print(data.tail())
print(data.describe())

x = data.drop([57],axis='columns')
y = data.iloc[:,-1]
print("*"*50)
print(y)
print("*"*50)
print("_"*50)
print(x)
print("_"*50)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 5, 10, 100, 300,500]
kernals = ['linear', 'poly', 'rbf']

dic = {'C' : [],
        'Linear' : [],
        'Poly' : [],
        'RBF' : [],
        }
for Cv in C_values:
  accuracies = []
  for ker in kernals:
    classifier = SVC(C=Cv, kernel=ker,degree=2, max_iter=-1)
    accuracy = calc_accu(x_train, y_train, x_test, y_test, classifier)
    accuracies.append(accuracy)
  dic['C'].append(Cv)
  dic['Linear'].append(accuracies[0])
  dic['Poly'].append(accuracies[1])
  dic['RBF'].append(accuracies[2])
  print( Cv, accuracies[0],accuracies[1],accuracies[2])
print(dic)
df = pd.DataFrame(dic)

pd.set_option('expand_frame_repr', False)
print(df)
df.to_csv("output_table.csv")