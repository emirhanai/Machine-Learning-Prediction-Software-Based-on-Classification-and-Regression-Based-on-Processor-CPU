import pandas as pd
import numpy as np
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.preprocessing import *
from sklearn.model_selection import *
from sklearn.metrics import *


data = pd.read_csv('data.csv')

X = data.drop(['Company','Processor Name'],axis='columns')
y = data.drop(['Turbo Speed (GHz)','Processor Name','Processor Cores','Processor Threads','Typical TDP (W)','Average CPU Mark'],axis='columns')

#load of change function for columns changing.
y_data = LabelEncoder()

#print(y)

y['Company_Change'] = y_data.fit_transform(y['Company'])

y_update_data = y.drop(['Company'],axis='columns')

float_y_update_data = np.float64(y_update_data)

#print(float_y_update_data)

#for i in np.arange(0,1,1):

#X_train,X_test,y_train and y_test files of creating (with suitable parameters).
X_train, X_test, y_train, y_test = train_test_split(X, y_update_data, test_size=0.2, random_state=15, shuffle=True,
                                                    stratify=None)
# model - processor classifier
model_processor = ExtraTreeClassifier(criterion="gini", splitter="random")

# model - processor regression
model_processor_regression = ExtraTreesRegressor(n_estimators=1)

# model - processor fit
model_processor_regression.fit(X_train, y_train)

# model - processor classifier fit
model_processor.fit(X_train, y_train)

# ""CLASSIFIER OF SCORE AND RESULT""

# model - processor classifier y_pred

y_pred_of_model = model_processor.predict(X_test)

# model classifier score of result
# print("Select of X {} ".format(i))
print("Classifier Accuracy Score: {} ".format(accuracy_score(y_test,y_pred_of_model)))
print("Classifier Precision Score: {} ".format(precision_score(y_test,y_pred_of_model)))
print("Classifier Recall Score: {} ".format(recall_score(y_test,y_pred_of_model)))
print("Classifier F1 Score: {} ".format(f1_score(y_test,y_pred_of_model)))
a,b,_ = roc_curve(y_test,y_pred_of_model)
print("Classifier AUC Score: {} ".format(auc(a,b)))
print("Classifier Confision Matrix: {} ".format(confusion_matrix(y_test,y_pred_of_model)))

# ""REGRESSION OF SCORE AND RESULT""

y_pred_of_regression_in_model = model_processor_regression.predict(X_test)

# print("Select of X {} ".format(i))
print("Regression Accuracy Score: {} ".format(accuracy_score(y_test, y_pred_of_regression_in_model)))
print("Regression Precision Score: {} ".format(precision_score(y_test, y_pred_of_regression_in_model)))
print("Regression Recall Score: {} ".format(recall_score(y_test, y_pred_of_regression_in_model)))
print("Regression F1 Score: {} ".format(f1_score(y_test, y_pred_of_regression_in_model)))
a, b, _ = roc_curve(y_test, y_pred_of_regression_in_model)
print("Regression AUC Score: {} ".format(auc(a, b)))
print("Regression Confision Matrix: {} ".format(confusion_matrix(y_test, y_pred_of_regression_in_model)))

# Enter you random value for Features :)
Processor_Cores = int(input("Enter, Processor Cores: "))
Processor_Threads = int(input("Enter, Processor Threads: "))
Turbo_Speed_GHz = float(input("Enter, Turbo Speed (GHz): "))
Typical_TDP_W = int(input("Enter, Typical TDP (W): "))
Average_CPU_Mark = int(input("Enter, Average CPU Mark: "))

# prediction, random value of Company!
prediction_of_company_random_value = model_processor_regression.predict(
    [[Processor_Cores, Processor_Threads, Turbo_Speed_GHz, Typical_TDP_W, Average_CPU_Mark]])

# I create of algorithm :)
data_class = pd.read_csv('class.csv', index_col=None, na_values=None)
class_value_detect = data_class.columns.values[int(prediction_of_company_random_value)]
print('Prediction company: {} '.format(class_value_detect))

# model classifier save of format to .dot file :)
from graphviz import Source
dotfile = open("emirhan_project.dot",'w')

graph_of_data_dot = Source(export_graphviz(model_processor,
 filled=True,
 rounded=True,
 out_file=dotfile,
 feature_names=X.columns,
 class_names=['AMD = 0','INTEL = 1']))
dotfile.close()

#CLASSIFICATION RESULT

#Classifier Accuracy Score: 1.0
#Classifier Precision Score: 1.0
#Classifier Recall Score: 1.0
#Classifier F1 Score: 1.0
#Classifier AUC Score: 1.0
#Classifier Confision Matrix: [[5 0]
                              #[0 2]]

#REGRESSION RESULT

#Regression Accuracy Score: 1.0
#Regression Precision Score: 1.0
#Regression Recall Score: 1.0
#Regression F1 Score: 1.0
#Regression AUC Score: 1.0
#Regression Confision Matrix: [[5 0]
                              #[0 2]]
