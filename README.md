# **Machine Learning Prediction Software Based on Classification and Regression Based on Processor [CPU] Specifications**

I developed Machine Learning Prediction Software based on Regression and Classifying of Processors [CPU] according to their specifications. This machine learning software passed all scores by 1.0, which is a big score. I used Extra tree classifier and Extra tree regression model. The processor company predictions according to the technical specifications you enter in the prediction tool.

The values you enter should be (respectively):

**1) Enter, Processor Cores:** 

**2) Enter, Processor Threads:** 

**3) Enter, Turbo Speed (GHz):**

**4) Enter, Typical TDP (W):** 

**5) Enter, Average CPU Mark:**


_Example:_ 

           `model_processor = ExtraTreeClassifier(criterion="gini", splitter="random")`
          
           `model_processor_regression = ExtraTreesRegressor(n_estimators=1)`

_Outpot :_ `Prediction company: AMD`

**I am happy to present this software to you!**

### CLASSIFICATION RESULT

`Classifier Accuracy Score: 1.0 
Classifier Precision Score: 1.0 
Classifier Recall Score: 1.0 
Classifier F1 Score: 1.0 
Classifier AUC Score: 1.0 
Classifier Confision Matrix: [[5 0]
                              [0 2]] `

### REGRESSION RESULT

`Regression Accuracy Score: 1.0 
Regression Precision Score: 1.0 
Regression Recall Score: 1.0 
Regression F1 Score: 1.0 
Regression AUC Score: 1.0 
Regression Confision Matrix: [[5 0]
                              [0 2]]` 


Data Source: [DataSource]

###**The coding language used:**

`Python 3.9.6`

###**Libraries Used:**

`Sklearn`

`Pandas`

`Numpy`

### **Developer Information:**

Name-Surname: **Emirhan BULUT**

Contact (Email) : **emirhan.bulut@turkiyeyapayzeka.com**

LinkedIn : **[https://www.linkedin.com/in/artificialintelligencebulut/][LinkedinAccount]**

[LinkedinAccount]: https://www.linkedin.com/in/artificialintelligencebulut/

Official Website: **[https://www.emirhanbulut.com.tr][OfficialWebSite]**

[OfficialWebSite]: https://www.emirhanbulut.com.tr

[DataSource]: https://www.cpubenchmark.net/desktop.html
