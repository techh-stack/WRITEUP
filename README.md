# ----------------------------------------------
# HEART DISEASE DATA OPERATIONS + CONFUSION MATRIX (with diagram)
# ----------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 2: Load Dataset
data = pd.read_csv("C:\Heart.csv")
print("First 5 rows of data:\n", data.head(), "\n")

# Step 3: Perform Data Operations
print("Shape of Data:", data.shape, "\n")
print("Missing Values in each column:\n", data.isnull().sum(), "\n")
print("Data Types:\n", data.dtypes, "\n")
print("Count of Zeroes in each column:\n", (data == 0).sum(), "\n")
print("Mean Age of Patients:", data['Age'].mean(), "\n")

# Step 4: Extract Selected Columns
selected_data = data[['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol']]
print("Selected Columns:\n", selected_data.head(), "\n")

# Step 5: Split Dataset (80% Training, 20% Testing)
train, test = train_test_split(selected_data, test_size=0.2, random_state=42)
print("Training Set Shape:", train.shape)
print("Testing Set Shape:", test.shape, "\n")

# Step 6: Confusion Matrix Example (COVID Test Case)
# Given Data:
# Predicted Positive = 100
# Actual Positive = 50
# True Positive (TP) = 45
# False Positive (FP) = 100 - 45 = 55
# False Negative (FN) = 50 - 45 = 5
# Total Samples = 500
# True Negative (TN) = 500 - (TP + FP + FN) = 395

# Create true and predicted values
y_true = [1]*50 + [0]*450        # 50 actual positives, 450 actual negatives
y_pred = [1]*45 + [0]*5 + [1]*55 + [0]*395  # predicted results

# Step 7: Confusion Matrix and Metrics
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm, "\n")

# Plot Confusion Matrix (Diagram)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix Diagram')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Step 8: Calculate and Display Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 Score :", round(f1, 3))

#new optional
tp, tn,fp,fn = 45,395,55,5
total = 500

acc= (tp+tn)/(total)
print(f'The  accuracy of model is:',acc)

prec= tp/(tp+fp)
print(f'The  precision of model is:',prec)

rec= tp/(tp+fn)
print(f'The  recall of model is:',rec)

f1_score = 2* (prec*rec)/(prec+rec)
print(f'The f1 score of model is:',f1_score)

# Step 9: Show only specific columns (e.g., Age and RestBP)
print("\nShowing only 'Age' and 'RestBP' columns:\n")
print(data[['Age', 'RestBP']].head())

print("\nShowing only 'Sex' and 'Chol' columns:\n")
print(data[['Sex','Chol']].head())
###################################################################################################################################
                                                                                2
######################################################################################################################################
# import required libraries are followiing:
import pandas as pd  # For data handling 
import numpy as np   # For numerical and mathematical operations 
import matplotlib.pyplot as plt  # For data visualization and plotting graphs 
from sklearn.linear_model import LinearRegression  # To perform Linear Regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
# For checking how well the model works

data = pd.read_csv('C:\Temperatures.csv')
# To load the dataset
data.head()
# Show the first few rows of the dataset
data.tail(n=9)
# Show the last 9 rows of the dataset
data.isnull().sum()
# Checks for missing values in the dataset

data.dtypes # Check what kind of data each column contains (numbers, text, dates, etc.)

data.describe() # Display average, min, max and other stats for the numeric data

data.info()

x = data[["YEAR"]] # Feature (independent variable)
y = data[["FEB"]] # Target (dependent variable)
#(we can change it as per our need for other months)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression() 
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# Show first 10 predictions vs actual 
print("\nSample Predictions (FAB):") 
print(pd.DataFrame({"Actual": y_test.values.flatten()
,"Predicted": y_test_pred.flatten()}).head(10))

train_mse = mean_squared_error(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\nPerformance Metrics for FAB month:")
print("Train = mean_squared_error:", train_mse, " mean_absolute_error:", train_mae," r2_score:", train_r2)

test_mse = mean_squared_error(y_test, y_test_pred) 
test_mae = mean_absolute_error(y_test, y_test_pred) 
test_r2 = r2_score(y_test, y_test_pred)

print("\nPerformance Metrics for FAB month:")
print("Test = mean_squared_error:", test_mse, " mean_absolute_error:", test_mae," r2_score:", test_r2)

# Plot training data and regression line
plt.scatter(X_train, y_train, color="blue", label="Train Data")  # Plot training data points
plt.plot(X_train, y_train_pred, color="yellow", label="Regression Line")  # Plot regression line for training data
plt.xlabel("Year")  # Set x-axis label
plt.ylabel("Temperature")  # Set y-axis label
plt.title("MAY Temperature vs Year (Training Set)")  # Set plot title for training data
plt.legend()  # Show legend
plt.show()  # Display training plot

# Plot testing data and regression line
plt.scatter(X_test, y_test, color="purple", label="Test Data")  # Plot testing data points
plt.plot(X_test, y_test_pred, color="orange", label="Regression Line")  # Plot regression line for testing data
plt.xlabel("Year") 
plt.ylabel("Temperature (°C)")  
plt.title("MAY Temperature VS Year (Test Set)")  # Set plot title for testing data
plt.legend() 
plt.show()  

import matplotlib.pyplot as plt  # Import matplotlib for plotting

sample = data.head(12)  # Take first 12 years as sample
plt.figure(figsize=(12,6))  # Set figure size

plt.bar(sample["YEAR"], sample["MAY"], color='pink', linewidth=6)  # Create bar chart of May temperatures
plt.xlabel("Year")  # Set x-axis label
plt.ylabel("Temperature")  # Set y-axis label
plt.title("May Temperatures over Years")  # Set plot title
plt.show()  # Display the plot


# Predict temperature for a specific year
year = 2095
x_new = np.array([[year]])  # Convert year to 2D array
predicted_temp = model.predict(x_new)  # Make prediction
print('Predicted Temperature for', year, ':', predicted_temp[0])  # Print result

# Plotting the waveform graph
plt.scatter(X_train, y_train, color='black', label="Actual")  # Plot actual training data points
plt.plot(X_test, model.predict(X_test), marker='o', color='blue', linewidth=3)  # Plot predicted values as line
plt.title("Predicted Temperature Over Years")  
plt.xlabel("Year")  
plt.ylabel("Predicted Temperature")  
plt.legend()  
plt.show()  


months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]  
# List of months to loop through

results = []  # Initialize an empty list to store performance metrics for each month

for m in months:
    X = data[["YEAR"]] # Select YEAR column as feature (independent variable)
    y = data[[m]] # Select the current month column as target (dependent variable)
    
    # Spliting data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression().fit(X_train, y_train)  # Training linear regression model on training data
    
    y_pred = model.predict(X_test)  # Predicting target values for the test set
    
    # Calculating model performance metrics
    mae = mean_absolute_error(y_test, y_pred)  # average absolute difference between actual and predicted
    mse = mean_squared_error(y_test, y_pred)   # average squared difference between actual and predicted
    r2 = r2_score(y_test, y_pred)             # proportion of variance explained by the model
    
    results.append([m, round(mae, 2), round(mse,2), round(r2,2)])  
    # Adding the month and metrics to results list

# Converting results list into a pandas DataFrame for better visualization
results_data = pd.DataFrame(results, columns=["Month", "MAE", "MSE", "R2"])  

print("\nPerformance Summary (All Months):")  # Printing a header
print(results_data)  # Displaying the performance metrics table
###################################################################################################################################
                                                                                3
#####################################################################################################################################
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load dataset
df = pd.read_csv("C:/Users/lenovo/Downloads/SMSSpamCollection", sep='\t', names=["label", "text"])

df
df.shape
df.head()
df.isnull().sum()
df['label'].value_counts() 

#pip install nltk

#import nltk

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

sent= 'How are you friend?'
word_tokenize(sent)

# --- a) Data Pre-processing ---
ps = PorterStemmer()
swords = set(stopwords.words('english'))

def clean_text(sent):
    tokens = word_tokenize(sent.lower())
    clean = [ps.stem(word) for word in tokens if word.isalpha() and word not in swords]
    return clean
    
sent = 'Hello friends! How are you? We willlearning python today.'
def clean_text(sent):
    tokens = word_tokenize(sent)
    clean = [word for word in tokens if word.isdigit() or word.isalpha()]
    clean = [ps.stem(word) for word in clean if word.lower() not in swords]
    return clean
    
clean_text(sent)

# Convert text to numerical features using TF-IDF
tfidf = TfidfVectorizer(analyzer=clean_text)
X = tfidf.fit_transform(df['text'])
y = df['label']

# --- b) Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

print(f"Size of splitted data")
print(f"X_train {X_train.shape}") 
print(f"y_train {y_train.shape}")
print(f"X_test {X_test.shape}")
print(f"y_test {y_test.shape}")

# --- c) Naive Bayes Model ---
nb = GaussianNB()
nb.fit(X_train.toarray(), y_train)
y_pred_nb = nb.predict(X_test.toarray())

y_test.value_counts()

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_nb)
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

# --- Random Forest Model ---
rf = RandomForestClassifier(random_state=1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf)
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Print detailed evaluation
print("\nNaive Bayes Report:\n", classification_report(y_test, y_pred_nb))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))

print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

import seaborn as sns
sns.countplot(x=y)
plt.title("Count of Ham vs Spam Messages")
###################################################################################################################################
                                                                                4
#####################################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\mall\Mall_Customers.csv")

df

x = df.iloc[:,3:]
x

df.shape

plt.title('Unclustered Data')
sns.scatterplot(x=x['Annual Income (k$)'],y=x['Spending Score (1-100)'])

from sklearn.cluster import KMeans, AgglomerativeClustering 
#here we applied 2 algom from package cluster

km = KMeans(n_clusters=4) #here we made 4 clusters

import os  # Imports the OS module to interact with the operating system
# Set the environment variable "OMP_NUM_THREADS" to "1"
# This limits the number of CPU threads used by libraries like NumPy or scikit-learn to 1
# It helps prevent excessive CPU usage or parallel processing issues
os.environ["OMP_NUM_THREADS"] = "1"

km.fit_predict(x)

km.inertia_ #sum of wcsse for (k=4) all the clusters 1, 2, 3 & 4. 

sse =[] # This is an array where the SSE value for each K will be recorded
for k in range(1,16):
    km = KMeans(n_clusters=k)
    km.fit_predict(x)
    sse.append(km.inertia_)
# Here, using the append function, we store that value in the array
# This means at index 0 (position 0), the SSE value for K=1 will be stored

sse

sns.lineplot(x=list(range(1,16)),y = sse)
plt.xlabel('Cluster')
plt.ylabel('SSE')

km = KMeans(n_clusters=5,random_state=100)

km_labels = km.fit_predict(x)

km.labels_

cent = km.cluster_centers_ 
#to extract the centroieds of clusters we call the method cluster_centers_ 
#from the km from the instacnce of the kmeans algom

cent

#creating plot for Clustered Data
plt.title('Clustered Data')
sns.scatterplot(x=x['Annual Income (k$)'],y=x['Spending Score (1-100)'],c=km_labels)
sns.scatterplot(x=cent[:,0],y=cent[:,1], s=300, color='black')

df[km_labels==4]

agl = AgglomerativeClustering(n_clusters=4)

alabels = agl.fit_predict(x)

alabels

plt.figure(figsize=(16,9))
plt.subplot(1,2,1)
plt.title('Agglomerative')
sns.scatterplot(x=x['Annual Income (k$)'],y=x['Spending Score (1-100)'], c= alabels) #km_labels)
#sns.scatterplot(x=cent[:,0],y=cent[:,1], s=200, color='red')
plt.subplot(1,2,2)
plt.title('KMEANS')
sns.scatterplot(x=x['Annual Income (k$)'],y=x['Spending Score (1-100)'],c=km_labels)
sns.scatterplot(x=cent[:,0],y=cent[:,1], s=200, color='red')

df['Cluster'] = km.labels_
print(df.head())
# Adds a new column showing each customer’s cluster group.

# Plotting only one specific cluster (Cluster 0)
cluster_0 = df[df['Cluster'] == 0]
plt.title('Cluster 0 Analysis')
plt.scatter(cluster_0['Annual Income (k$)'], cluster_0['Spending Score (1-100)'], color='blue', label='Cluster 0')
# Plot only the centroid of Cluster 0
plt.scatter(
    cent[0, 0],  # x-coordinate of cluster 0 centroid
    cent[0, 1],  # y-coordinate of cluster 0 centroid
    s=300,
    color='red',
    label='Centroid (Cluster 0)',
    marker='X'
)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
# Helps to analyze one group (Cluster 0) separately with its own centroid.
###################################################################################################################################
                                                                                55555
#####################################################################################################################################
# Step a) Data Pre-processing
# ---------------------------------------------------------

# Install mlxtend if not already installed
# !pip install mlxtend

# Import necessary libraries
import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset (CSV file should be in the same folder)
data = []
with open("C:/Users/lenovo/Desktop/Market_Basket_Optimisation.csv") as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        data.append(row)

# To load the dataset
df=pd.DataFrame(x, columns = te.columns_)
df.head(n=3)
# Show the first few rows of the dataset
df.tail(n=9)
# Show the last 9 rows of the dataset
df.isnull().sum()
# Checks for missing values in the dataset

df.dtypes # Check what kind of data each column contains (numbers, text, dates, etc.)

df.describe() # Display average, min, max and other stats for the numeric data

df.info()

# Display a few transactions to understand the structure
print("Sample Transactions:\n", data[1:10]) 

# Check total number of transactions
print("Total transactions:", len(data))

# ---------------------------------------------------------
# Step b) Generate the list of transactions from the dataset
# ---------------------------------------------------------

# Convert the list of transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
x = te.fit_transform(data)

x

te.columns_

# Convert the encoded data to a pandas DataFrame
df = pd.DataFrame(x, columns=te.columns_)

# Display first few rows of the DataFrame
print("\nEncoded Transaction DataFrame:\n")
print(df.head())

# ---------------------------------------------------------
# Step c) Train Apriori algorithm on the dataset
# ---------------------------------------------------------

# Find frequent itemsets with a minimum support threshold
freq_itemset = apriori(df, min_support=0.01, use_colnames=True)

# Display frequent itemsets
print("\nFrequent Itemsets:\n")
print(freq_itemset.head())

freq_itemset

# Generate association rules based on the frequent itemsets
rules = association_rules(freq_itemset, metric='confidence', min_threshold=0.10)

rules

# Select key columns for clarity
rules = rules[['antecedents', 'consequents', 'support', 'confidence']]

# Display the rules
print("\nAssociation Rules:\n")
print(rules.head(10))

# ---------------------------------------------------------
# Step d) Visualize the list of rules
# ---------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs Confidence (size by lift)')
#plt.grid(True)
plt.show()
