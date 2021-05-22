# Behavior-of-Linear-Model-Machine-Learning-Study
experimenting on a linear model to study behavior on  different data set

# 1. Experiment 1 : Imbalanced Dataset 

## 1.1 Linear SVM 

 1.1.1  As a part of this experiment we will observe how linear models work in case of data imbalanced, we will observe how hyper plane is changes according to change in  learning rate for both svm and logistic regression.
    
1.1.2    we have created 4 random datasets which are linearly separable and having class imbalance

1.1.3 In the  dataset the ratio between positive and negative is 100 : 2, 100 : 20, 100 : 40, 100 : 80.

![SVM : 100:2, 100:20 ](https://i.imgur.com/nLJYwdF.png)

![SVM : 100:40, 100:80 ](https://i.imgur.com/KZngsMS.png)


## 1.2 Linear SVM Observation
1.2.1
<pre>

For C = 0.001
for Dataset 1 (100:2) model is under-fitted
for Dataset 2 (100:20) model is under-fitted than dataset 1
for Dataset 3 (100:40) model is more under-fitted than dataset 2
for Dataset 4 (100:80) model is by far most under-fitted
</pre>

1.2.2
<pre>
For C = 1
for Dataset 1 (100:2) model is under-fitted
for Dataset 2 (100:20) model is under-fitted than dataset 1
for Dataset 3 (100:40) model is more under-fitted than dataset 2
for Dataset 4 (100:80) model is highly under-fitted
</pre>
1.2.3
<pre>
For C = 100
for Dataset 1 (100:2) model is slightly under-fitted, but better than previous c =0.01 and c=1
for Dataset 2 (100:20) model is fitted well
for Dataset 3 (100:40) model is very slightly over-fitted
for Dataset 4 (100:80) model is fitted well
</pre>

