
##Data Collection
from sklearn.datasets import load_iris
#load iris dataset
iris=load_iris()
#print the feature data(X) and target(y)
print(iris.data)
print(iris.target)

##Data Preprocessing
from sklearn.model_selection import train_test_split
#split dataset into features and target
X=iris.data
y=iris.target
#split dataset into training set and testing set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

##Exploratory Data Analysis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#convert iris dataset into a Dataframe
iris_df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['species']=iris.target_names[iris.target]
#visualize the distribution of each feature
sns.pairplot(data=iris_df,hue='species')
plt.show()

##Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
#create a decision tree classifier
clf=DecisionTreeClassifier(random_state=42)
#train the model on training data
clf.fit(X_train,y_train)

##Model Evaluation
from sklearn.metrics import accuracy_score
#make predictions on testing data
y_pred=clf.predict(X_test)
#calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.2f}")

##Code to use the model
import joblib
#save the model to a file
joblib.dump(clf,'decision_tree_model.joblib')

##Using the model
#load the trained model
model=joblib.load('decision_tree_model.joblib')

##Code to take input from the user
#define a dictionary to map numerical labels to species names
species_mapping={0:'setosa',1:'versicolor',2:'virginica'}
#preprocessing user input
def preprocess_user_input():
    while True:
        try:
            sl=float(input("Enter sepal length(cm):: "))
            if 4.0<=sl<=8.0:
                break
            else:
                print("Invalid input.Sepal lenght must be between4.0 and 8.0")
        except ValueError:
            print("Invalid input. Please enter a numeric value for sepal length.")
    while True:
        try:
            sw = float(input("Enter sepal width (cm): "))
            if 2.0 <= sw <= 4.5:
                break
            else:
                print("Invalid input. Sepal width must be between 2.0 and 4.5.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for sepal width.")
    
    while True:
        try:
            pl = float(input("Enter petal length (cm): "))
            if 1.0 <= pl <= 7.0:
                break
            else:
                print("Invalid input. Petal length must be between 1.0 and 7.0.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for petal length.")
    
    while True:
        try:
            pw = float(input("Enter petal width (cm): "))
            if 0.1 <= pw <= 2.5:
                break
            else:
                print("Invalid input. Petal width must be between 0.1 and 2.5.")
        except ValueError:
            print("Invalid input. Please enter a numeric value for petal width.")
    return [sl,sw,pl,pw]      


def get_user_input():
    num_cases = int(input("Enter the number of cases to predict: "))
    predictions = []

    for i in range(num_cases):
        print(f"Case {i+1}:")
        
        user_input = preprocess_user_input()
        predictions.append(user_input)
    
    return predictions

##Code to make predictions on the given user input
# Collect user input for multiple cases
user_input_list = get_user_input()

# Make predictions for each case and display the predicted class
for i, user_input in enumerate(user_input_list):
    predicted_class = model.predict([user_input])[0]#get the numerical label
    predicted_species=species_mapping[predicted_class]#map to species name
    print(f"Predicted Iris Species for Case {i+1}: {predicted_species}")
