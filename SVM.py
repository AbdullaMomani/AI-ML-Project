import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the training and testing data
train_data_path = 'C:\\Users\\Abdullah\\Desktop\\Training_Data.xlsx'
test_data_path = 'C:\\Users\\Abdullah\\Desktop\\Testing_Data.xlsx'

train_df = pd.read_excel(train_data_path)
test_df = pd.read_excel(test_data_path)

# Define the target column
target_column = 'fetal_health'  # Assuming 'fetal_health' is the name of the target column

# Split features and target variable
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]
X_test = test_df.drop(columns=[target_column])
y_test = test_df[target_column]

# Handle categorical variables (if any) - Assuming no categorical variables as per description
# Example: encode categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_columns:
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# Feature scaling (standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Suspect', 'Pathological'])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Suspect', 'Pathological'], yticklabels=['Normal', 'Suspect', 'Pathological'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
