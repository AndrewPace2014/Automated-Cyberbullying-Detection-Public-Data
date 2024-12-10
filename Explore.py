import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Load the dataset
def load_csv(file_path):
    if os.path.exists(file_path):
        print(f"Loading {file_path}")
        df = pd.read_csv(file_path)
        print(f"Columns in {file_path}:")
        print(df.dtypes)
        return df
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

labels_data = load_csv('A Comprehensive Dataset for Automated Cyberbullying Detection/6. CB_Labels.csv')

# Prepare the data
required_columns = ['Total_messages', 'Aggressive_Count', 'Intent_to_Harm', 'Peerness']
X = labels_data[required_columns]
y = labels_data['CB_Label']

# Handle missing values by filling them with zero
X = X.fillna(0)

# Check the balance of the dataset
print("Class distribution in y:")
print(y.value_counts())

# Split the data into training and testing sets
print("Splitting the data into training and testing sets")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42, shuffle=True, stratify=y)

# Standardize the data
print("Standardizing the data")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the best configuration based on previous results
best_config = {'hidden_layer_sizes': (32, 16), 'activation': 'relu', 'solver': 'adam'}

print(f"\nUsing the best configuration: {best_config}")
model = MLPClassifier(hidden_layer_sizes=best_config['hidden_layer_sizes'], activation=best_config['activation'], solver=best_config['solver'], max_iter=500)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')

# Generate the confusion matrix
print("\nGenerating the confusion matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Generate a classification report
print("\nGenerating the classification report")
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)

# Extract feature importances using permutation importance
print("\nExtracting feature importances using permutation importance")
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
importance_means = result.importances_mean
for feature, importance in zip(required_columns, importance_means):
    print(f'Feature: {feature}, Importance: {importance:.4f}')

# Plot the loss curve
print("\nPlotting the loss curve")
plt.plot(model.loss_curve_)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()