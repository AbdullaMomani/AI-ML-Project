import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#Step 2: Load the dataset
file_path = r'C:\\Users\\Abdullah\Desktop\\fetal_health_Dataset.xlsx'
df = pd.read_excel(file_path)

#Step 3: Preprocess the data if needed (handle missing values, encode categorical variables, scale numerical features)

#Step 4: Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['fetal_health'])  #Features
y = df['fetal_health']  #Target variable

#Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Step 6: Choose a machine learning algorithm and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Step 7: Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#fine-tune the model if necessary (hyperparameter tuning)
