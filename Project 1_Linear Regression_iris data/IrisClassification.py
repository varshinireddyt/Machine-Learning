import numpy as np
import pandas as pd

iris_df = pd.read_csv('/Users/srikanthadavalli/PycharmProjects/Project1/iris.data')
iris_df.columns = ["Sepal Length","Sepal Width", "Petal Length", "Petal Width", "Species"]
iris_df = iris_df.replace({'Species' : {'Iris-setosa' : 1, 'Iris-versicolor' : 2,'Iris-virginica' : 3}})
iris_df.describe()
iris_array = iris_df[["Sepal Length","Sepal Width", "Petal Length", "Petal Width", "Species"]].values
#converting the dataframe into array
X = np.insert(iris_array,0,1,axis = 1)
X = np.delete(X,-1,axis=1)
Y = iris_array[:,-1]
#Finding Beta
def findBeta(X, Y):
    X_transpose = np.transpose(X)
    X_transpose_dot = np.dot(X_transpose, X)
    X_transpose_inverse = np.linalg.inv(X_transpose_dot)
    X_inverse_dot = np.dot(X_transpose_inverse, X_transpose)
    beta = np.dot(X_inverse_dot, Y)
    return beta
#Cross Fold Valdiation and Predicting Y values
def findCV(K):
    #calculating fold
    n = len(X)
    fold = n//K
    #splitting the data based on the fold
    X_fold = np.array_split(X,K)
    Y_fold = np.array_split(Y,K)

    #train and test sets
    for i in range(K):
        test_X, test_Y = np.array(X_fold[i]), np.array(Y_fold[i])
        train_X, train_Y = np.delete(X_fold,i,axis= 0),np.delete(Y_fold,i,axis = 0)
        train_X,train_Y = np.concatenate(train_X),np.concatenate(train_Y)
        """
        Linear regression :
         f(x) = beta . X
         where X is the test set
        """
        y_pred =  np.dot(test_X, findBeta(train_X,train_Y))
        accuracy = findAccuracy(y_pred, test_Y,fold)
        return accuracy
#Finding the accuracy
def findAccuracy(y_pred, test_Y, fold):
        count = 0
        accuracy_list = []
        for j in np.equal(test_Y, y_pred.round()):
            if j == True:
                count+=1
        accuracy = count/fold
        accuracy_list.append(accuracy)
        return((sum(accuracy_list)/len(accuracy_list)))


print("The model predicted values of Iris dataset with an overall accuracy of %s: " % findCV(2) )
print("The model predicted values of Iris dataset with an overall accuracy of %s: " % findCV(5) )
print("The model predicted values of Iris dataset with an overall accuracy of %s: " % findCV(10) )
print("The model predicted values of Iris dataset with an overall accuracy of %s: " % findCV(15) )
print("The model predicted values of Iris dataset with an overall accuracy of %s: " % findCV(20) )

