# Imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# MODEL
def train_and_save_stacking_model(data_path, stacking_model_path):
    df = pd.read_excel(data_path)

    # Calculate emp_score
    df['emp_score'] = 0.61 * df['ssc_p'] + 0.49 * df['hsc_p'] + 0.48 * df['degree_p'] + 0.28 * df['workex']

    # Splitting the data
    x = df[['etest_p', 'emp_score', 'mba_p']]
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10)
    
    LR = LogisticRegression()
    DT = DecisionTreeClassifier(random_state=0)
    RF = RandomForestClassifier(max_depth=5, random_state=0)
    svc = SVC()
    
    # Stacking model for classification
    estimators = [('DT', DT), ('RF', RF), ('SVC', svc)]
    stack_model = StackingClassifier(estimators=estimators, final_estimator=LR)
    
    stack_model.fit(X_train, y_train)
    
    # Save the stacking model using joblib
    joblib.dump(stack_model, stacking_model_path)

def train_and_save_ridge_model(data_path, ridge_model_path):
    df = pd.read_excel(data_path)

    columns = ['status']
    df = df.replace(0, np.nan).dropna(axis=0, how='any', subset=columns).fillna(0).astype(int)

    df['emp_score_for_sal'] = 0.54 * df['ssc_p'] + 0.45 * df['hsc_p'] + 0.41 * df['degree_p'] #+ 0.3 * df['workex']
    
    X = df[['etest_p', 'emp_score_for_sal', 'mba_p']]
    y = df['salary']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = Ridge(alpha=100.0)
    model.fit(X_train, y_train)
    
    joblib.dump(model, ridge_model_path)

def predict_status_with_stacking(input_dict, stacking_model_path):
    # Load the saved stacking model
    stacking_model = joblib.load(stacking_model_path)

    # Create a DataFrame from the input dictionary
    df = pd.DataFrame([input_dict])

    # Calculate emp_score
    df['emp_score'] = 0.61 * df['ssc_p'] + 0.49 * df['hsc_p'] + 0.48 * df['degree_p'] + 0.28 * df['workex']

    # Select relevant features
    x = df[['etest_p', 'emp_score', 'mba_p']]

    # Predict using the stacking model
    status_pred = stacking_model.predict(x)
    return status_pred[0]  # Return the predicted status

def predict_salary_range(input_dict, ridge_model_path):
    model = joblib.load(ridge_model_path)

    # Calculate the emp_score_for_sal value based on the input dictionary
    emp_score_for_sal = 0.19 * input_dict['ssc_p'] + 0.19 * input_dict['hsc_p'] + \
                        0.18 * input_dict['degree_p'] + 0.28 * input_dict['workex']

    # Predict using the model
    predicted_salary = model.predict([[input_dict['etest_p'], emp_score_for_sal, input_dict['mba_p']]])[0]

    # Rounding to get a range
    range_width = 50000
    lower_bound = int(predicted_salary // range_width) * range_width
    upper_bound = lower_bound + range_width
    return (lower_bound, upper_bound)
 

# DATA PATHS
data_file_path = "C:\\Users\\sneha\\OneDrive - SSN Trust\\SSN\\IBM Hack Challenge\\IBM Processed Data.xlsx"
stacking_model_path = 'stacking_model.pkl'
ridge_model_path = 'trained_ridge_model.pkl'

# Training the models
train_and_save_stacking_model(data_file_path, stacking_model_path)
train_and_save_ridge_model(data_file_path, ridge_model_path)  


# Function to create input data from the User
def create_input_data(form_data):
    input_data = {
        'ssc_p': float(form_data["ssc_p"]),
        'hsc_p': float(form_data["hsc_p"]),
        'degree_p': float(form_data["degree_p"]),
        'workex': int(form_data["workex"]),
        'etest_p': float(form_data["etest_p"]),
        'mba_p': float(form_data["mba_p"])
        }
    return input_data

input_data = 0

# Predict the status using the stacking model
def predict_placement(input_data, stacking_model_path, ridge_model_path):
    predicted_status = predict_status_with_stacking(input_data, stacking_model_path)
    return predicted_status

def call_predict_placement(form_data):
    input_data = create_input_data(form_data)
    predicted_status = predict_placement(input_data, stacking_model_path, ridge_model_path)
    if (predicted_status == 1):
        # Predict the salary range using the Ridge model
        predicted_range = predict_salary_range(input_data, ridge_model_path)
        return {"status": "yes", "lower_bound": predicted_range[0], "upper_bound": predicted_range[1]}
          
    else:
        return


""""
if (predicted_status == 1):
    print("Placed")
    # Predict the salary range using the Ridge model
    predicted_range = predict_salary_range(input_data, ridge_model_path)
    print("Salary: ",predicted_range)   
else:
    print("Not Placed")
"""
