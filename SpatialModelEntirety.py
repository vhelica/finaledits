# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 09:50:24 2025

@author: 15759
"""

#file1
#get all vars and elevation

import rasterio
import numpy as np
import pandas as pd
import os
from rasterio.transform import rowcol

# Define directories for all climate variables
tmax_dir = "wc2.1_30s_tmax"
tmin_dir = "wc2.1_30s_tmin"
tavg_dir = "wc2.1_30s_tavg"
wind_dir = "wc2.1_30s_wind"
prec_dir = "wc2.1_30s_prec"
vapr_dir = "wc2.1_30s_vapr"
srad_dir = "wc2.1_30s_srad"
elev_file = "wc2.1_30s_elev.tif"  # Single file for elevation

# List of all climate variables to process
variables = {
    'tmax': tmax_dir,
    'tmin': tmin_dir,
    'tavg': tavg_dir,
    'wind': wind_dir,
    'prec': prec_dir,
    'vapr': vapr_dir,
    'srad': srad_dir,
    'elev': None  # Elevation is a single file, not a directory
}

# Generate file paths for each month (1-12) for all variables
file_paths = {}
for var, dir_name in variables.items():
    if var == 'elev':  # Special handling for elevation (single file)
        file_paths[var] = [elev_file]
    else:  # Monthly files for climate variables
        file_paths[var] = [os.path.join(dir_name, f"wc2.1_30s_{var}_{month:02d}.tif") for month in range(1, 13)]

# Load coordinates from the combined_data file
combined_data = pd.read_csv("VGUniversalDatasetCSV.csv")

# Ensure we have longitude and latitude columns
if 'Longitude' not in combined_data.columns or 'Latitude' not in combined_data.columns:
    print("Error: combined_data file must contain 'Longitude' and 'Latitude' columns")
    exit(1)

# Create new columns for all climate variables and elevation
for var in variables.keys():
    combined_data[f"{var.capitalize()}"] = np.nan

# Function to extract values from files for a given coordinate
def extract_values(files, lon, lat):
    values = []
    
    for file in files:
        try:
            with rasterio.open(file) as src:
                # Convert geographic coordinates to raster row/col indices
                row_idx, col_idx = rowcol(src.transform, lon, lat)
                
                # Check if the point is within the raster bounds
                if 0 <= row_idx < src.height and 0 <= col_idx < src.width:
                    # Read only the single pixel value
                    window = ((row_idx, row_idx+1), (col_idx, col_idx+1))
                    value = src.read(1, window=window)[0][0]
                    
                    # Store the value without scaling
                    if value != src.nodata:
                        values.append(float(value))
        except Exception as e:
            print(f"Error processing {file} for point ({lon}, {lat}): {e}")
    
    return values

# Process each coordinate point
total_points = len(combined_data)
for idx, row in combined_data.iterrows():
    lon, lat = row['Longitude'], row['Latitude']
    
    # Extract values for each climate variable and elevation
    for var in variables.keys():
        values = extract_values(file_paths[var], lon, lat)
        
        # Calculate and store annual averages (for climate variables) or the single value (for elevation)
        if values:
            if var == 'elev':  # Elevation is a single value
                combined_data.at[idx, f"{var.capitalize()}"] = values[0]
            else:  # Climate variables: average over months
                combined_data.at[idx, f"{var.capitalize()}"] = np.mean(values)
    
    # Print progress
    if (idx + 1) % 10 == 0 or idx == total_points - 1:
        print(f"Processed {idx + 1}/{total_points} points ({(idx + 1) / total_points * 100:.1f}%)")

# Save the updated data
combined_data.to_csv("combined_data_with_climate_and_elev.csv", index=False)
print("Results saved to 'combined_data_with_climate_and_elev.csv'")

# Print a summary
print("\nSummary of data retrieval:")
for var in variables.keys():
    valid_count = combined_data[f"{var.capitalize()}"].notna().sum()
    print(f"  - {var.capitalize()}: {valid_count}/{total_points} points ({valid_count/total_points*100:.1f}%)")

#############################################################
#file 2
#big compare final

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:48:17 2025

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 13:15:45 2025

@author: Ashley
"""
#file2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.feature_selection import RFECV
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
from sklearn.inspection import PartialDependenceDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
import warnings


# Suppress the specific warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, 
                       message="X has feature names, but .* was fitted without feature names")
# Load and prepare the data
df = pd.read_csv("combined_data_with_climate_and_elev_hand_mod.csv")

# Convert the binary response variable to numeric
label_encoder = LabelEncoder()
df['Parity_encoded'] = label_encoder.fit_transform(df['Parity'])  # V and O become 0 and 1
print(f"Encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Drop rows with missing values or handle them appropriately
df = df.dropna()
print(f"Shape after dropping missing values: {df.shape}")

# Split the data into features and target
X = df.drop(['Parity', 'Parity_encoded', 'Source', 'Latitude', 'Longitude'], axis=1)  # Assuming 'Source' is metadata
y = df['Parity_encoded']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for naming features
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Prepare a results dictionary to store all model metrics
results = {}

# ------------------------------------------------------------
# A) Linear Model (Logistic Regression) with Stepwise Selection
# ------------------------------------------------------------
print("\n--- Fitting Logistic Regression with Interaction Terms ---")

# Base formula with all main effects
base_formula = "Parity_encoded ~ " + " + ".join(X_train.columns)

# Create formulas with interaction terms
interaction_formulas = []
for i, (var1, var2) in enumerate(combinations(X_train.columns, 2)):
    interaction_formulas.append(f"{base_formula} + {var1}:{var2}")

# Fit statsmodels logistic regression for each formula
logit_models = []
for formula in [base_formula] + interaction_formulas:
    model = smf.logit(formula=formula, data=pd.concat([X_train.reset_index(drop=True), 
                                                     pd.DataFrame({'Parity_encoded': y_train}).reset_index(drop=True)], 
                                                    axis=1)).fit(disp=False)
    logit_models.append((formula, model))

# Find best model by AIC
best_model_idx = np.argmin([model.aic for _, model in logit_models])
best_formula, best_logit = logit_models[best_model_idx]

print(f"Best Logistic Model Formula by AIC:\n{best_formula}")
print(f"AIC: {best_logit.aic:.2f}")

# Predictions and evaluation for best logistic model
y_pred_prob_logit = best_logit.predict(pd.concat([X_test.reset_index(drop=True), 
                                                pd.Series(y_test).reset_index(drop=True)], axis=1))
y_pred_logit = (y_pred_prob_logit > 0.5).astype(int)

# Calculate metrics
accuracy_logit = accuracy_score(y_test, y_pred_logit)
report_logit = classification_report(y_test, y_pred_logit)
conf_matrix_logit = confusion_matrix(y_test, y_pred_logit)

# ROC curve
fpr_logit, tpr_logit, _ = roc_curve(y_test, y_pred_prob_logit)
roc_auc_logit = auc(fpr_logit, tpr_logit)

# Variable importance from coefficients
coef_importance = pd.DataFrame({
    'Variable': best_logit.params.index[1:],  # Skip intercept
    'Coefficient': best_logit.params.values[1:],
    'Std_Error': best_logit.bse.values[1:],
    'P_Value': best_logit.pvalues.values[1:],
    'Abs_Importance': np.abs(best_logit.params.values[1:])
}).sort_values('Abs_Importance', ascending=False)

print("\nLogistic Regression - Variable Importance:")
print(coef_importance[['Variable', 'Coefficient', 'P_Value']].head(10))

# Store results
results['Logistic'] = {
    'accuracy': accuracy_logit,
    'roc_auc': roc_auc_logit,
    'report': report_logit,
    'conf_matrix': conf_matrix_logit,
    'importance': coef_importance,
    'model': best_logit,
    'y_pred': y_pred_logit,
    'y_prob': y_pred_prob_logit
}


# ---------------------------------------------------------
# A2) Logistic Regression with Polynomial Features
# ---------------------------------------------------------
print("\n--- Fitting Logistic Regression with Polynomial Features ---")


# Create polynomial features (quadratic terms)
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Get feature names for polynomial terms
poly_feature_names = []
for i, name in enumerate(poly.get_feature_names_out(X_train.columns)):
    poly_feature_names.append(name.replace(' ', '*'))

# Fit logistic regression with L1 regularization to handle the large number of features
logit_poly = LogisticRegressionCV(
    cv=5, 
    penalty='l1', 
    solver='liblinear',
    random_state=42,
    max_iter=1000
)
logit_poly.fit(X_train_poly, y_train)

# Predictions
y_pred_prob_logit_poly = logit_poly.predict_proba(X_test_poly)[:, 1]
y_pred_logit_poly = logit_poly.predict(X_test_poly)

# Calculate metrics
accuracy_logit_poly = accuracy_score(y_test, y_pred_logit_poly)
report_logit_poly = classification_report(y_test, y_pred_logit_poly)
conf_matrix_logit_poly = confusion_matrix(y_test, y_pred_logit_poly)

# ROC curve
fpr_logit_poly, tpr_logit_poly, _ = roc_curve(y_test, y_pred_prob_logit_poly)
roc_auc_logit_poly = auc(fpr_logit_poly, tpr_logit_poly)

print(f"Accuracy: {accuracy_logit_poly:.4f}")
print(f"ROC AUC: {roc_auc_logit_poly:.4f}")
print(report_logit_poly)

# Variable importance from coefficients
coef_importance_poly = pd.DataFrame({
    'Variable': poly_feature_names,
    'Coefficient': logit_poly.coef_[0],
    'Abs_Importance': np.abs(logit_poly.coef_[0])
}).sort_values('Abs_Importance', ascending=False)

print("\nPolynomial Logistic Regression - Top 10 Variable Importance:")
print(coef_importance_poly[['Variable', 'Coefficient']].head(10))

# Store results
results['Logistic_Poly'] = {
    'accuracy': accuracy_logit_poly,
    'roc_auc': roc_auc_logit_poly,
    'report': report_logit_poly,
    'conf_matrix': conf_matrix_logit_poly,
    'importance': coef_importance_poly,
    'model': logit_poly,
    'y_pred': y_pred_logit_poly,
    'y_prob': y_pred_prob_logit_poly
}

# ----------------------------
# B) Random Forest Classifier
# ----------------------------
print("\n--- Fitting Random Forest Classifier ---")

# Parameter grid for random forest
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Grid search with cross-validation
rf = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc')
grid_search_rf.fit(X_train_scaled, y_train)

# Best model
rf_best = grid_search_rf.best_estimator_
print(f"Best Random Forest Parameters: {grid_search_rf.best_params_}")

# Predictions
y_pred_prob_rf = rf_best.predict_proba(X_test_scaled)[:, 1]
y_pred_rf = rf_best.predict(X_test_scaled)

# Metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# ROC curve
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Feature importance
feature_importance_rf = pd.DataFrame({
    'Variable': X_train.columns,
    'Importance': rf_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest - Variable Importance:")
print(feature_importance_rf.head(10))

# Store results
results['RandomForest'] = {
    'accuracy': accuracy_rf,
    'roc_auc': roc_auc_rf,
    'report': report_rf,
    'conf_matrix': conf_matrix_rf,
    'importance': feature_importance_rf,
    'model': rf_best,
    'y_pred': y_pred_rf,
    'y_prob': y_pred_prob_rf
}

# -----------------------------
# C) Gradient Boosting Classifier
# -----------------------------
print("\n--- Fitting Gradient Boosting Classifier ---")

# Parameter grid for gradient boosting
param_grid_gb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 7, 10],  # Allow deeper splits
    'min_samples_split': [2, 5],
    'subsample': [0.8, 1.0]
}

# Grid search with cross-validation
gb = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=5, scoring='roc_auc')
grid_search_gb.fit(X_train_scaled, y_train)

# Best model
gb_best = grid_search_gb.best_estimator_
print(f"Best Gradient Boosting Parameters: {grid_search_gb.best_params_}")

# Predictions
y_pred_prob_gb = gb_best.predict_proba(X_test_scaled)[:, 1]
y_pred_gb = gb_best.predict(X_test_scaled)

# Metrics
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)
conf_matrix_gb = confusion_matrix(y_test, y_pred_gb)

# ROC curve
fpr_gb, tpr_gb, _ = roc_curve(y_test, y_pred_prob_gb)
roc_auc_gb = auc(fpr_gb, tpr_gb)

# Feature importance
feature_importance_gb = pd.DataFrame({
    'Variable': X_train.columns,
    'Importance': gb_best.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nGradient Boosting - Variable Importance:")
print(feature_importance_gb.head(10))

# Store results
results['GradientBoosting'] = {
    'accuracy': accuracy_gb,
    'roc_auc': roc_auc_gb,
    'report': report_gb,
    'conf_matrix': conf_matrix_gb,
    'importance': feature_importance_gb,
    'model': gb_best,
    'y_pred': y_pred_gb,
    'y_prob': y_pred_prob_gb
}

# -----------------------------
# D) Neural Network Classifier
# -----------------------------
print("\n--- Fitting Neural Network Classifier ---")

# First attempt with standard GridSearch
print("Initial neural network training...")
param_grid_nn = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
}

# Grid search with cross-validation
nn = MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)
grid_search_nn = GridSearchCV(nn, param_grid_nn, cv=5, scoring='roc_auc')
grid_search_nn.fit(X_train_scaled, y_train)

# Best model from initial search
nn_best_initial = grid_search_nn.best_estimator_
print(f"Best initial neural network parameters: {grid_search_nn.best_params_}")

# Now create a more comprehensive model with extended training
print("\nTraining extended neural network model...")
# Get best parameters from initial search
best_hidden_layers = grid_search_nn.best_params_['hidden_layer_sizes']
best_activation = grid_search_nn.best_params_['activation']
best_alpha = grid_search_nn.best_params_['alpha']
best_learning_rate = grid_search_nn.best_params_['learning_rate_init']

# Create a more complex model based on the best parameters
nn_extended = MLPClassifier(
    hidden_layer_sizes=best_hidden_layers,
    activation=best_activation,
    alpha=best_alpha,
    learning_rate_init=best_learning_rate * 0.5,  # Slightly lower learning rate for stability
    max_iter=500,  # Increased epochs
    random_state=42,
    early_stopping=False,  # We want it to run the full 500 epochs
    verbose=False,  # Show training progress
    learning_rate='adaptive',  # Adaptive learning rate for better convergence
    n_iter_no_change=50,  # More patience
    tol=1e-5  # Tighter tolerance
)

# Fit the extended model
nn_extended.fit(X_train_scaled, y_train)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(nn_extended.loss_curve_)
plt.title('Neural Network Learning Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('nn_learning_curve.png')
plt.close()

# Predictions using the extended model
y_pred_prob_nn = nn_extended.predict_proba(X_test_scaled)[:, 1]
y_pred_nn = nn_extended.predict(X_test_scaled)

# Metrics
accuracy_nn = accuracy_score(y_test, y_pred_nn)
report_nn = classification_report(y_test, y_pred_nn)
conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)

# ROC curve
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_prob_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

print(f"\nNeural Network Performance:")
print(f"Accuracy: {accuracy_nn:.4f}")
print(f"ROC AUC: {roc_auc_nn:.4f}")
print(report_nn)

# Also try a larger network with deeper architecture
print("\nTraining deeper neural network model...")
nn_deep = MLPClassifier(
    hidden_layer_sizes=(100, 50, 25),  # Deeper architecture
    activation=best_activation,
    alpha=best_alpha,
    learning_rate_init=best_learning_rate * 0.5,
    max_iter=500,
    random_state=42,
    early_stopping=False,
    verbose=False,
    learning_rate='adaptive',
    n_iter_no_change=50,
    tol=1e-5
)

nn_deep.fit(X_train_scaled, y_train)

# Predictions using the deep model
y_pred_prob_nn_deep = nn_deep.predict_proba(X_test_scaled)[:, 1]
y_pred_nn_deep = nn_deep.predict(X_test_scaled)

# Metrics for deep model
accuracy_nn_deep = accuracy_score(y_test, y_pred_nn_deep)
roc_auc_nn_deep = auc(*roc_curve(y_test, y_pred_prob_nn_deep)[:2])

print(f"\nDeeper Neural Network Performance:")
print(f"Accuracy: {accuracy_nn_deep:.4f}")
print(f"ROC AUC: {roc_auc_nn_deep:.4f}")

# Choose the better model based on ROC AUC
if roc_auc_nn_deep > roc_auc_nn:
    print("Using the deeper neural network model as it performed better.")
    nn_best = nn_deep
    y_pred_nn = y_pred_nn_deep
    y_pred_prob_nn = y_pred_prob_nn_deep
    accuracy_nn = accuracy_nn_deep
    roc_auc_nn = roc_auc_nn_deep
    # Recalculate metrics for the better model
    report_nn = classification_report(y_test, y_pred_nn)
    conf_matrix_nn = confusion_matrix(y_test, y_pred_nn)
else:
    print("Using the extended neural network model as it performed better.")
    nn_best = nn_extended

# Feature importance via permutation importance
print("\nCalculating feature importance via permutation (this may take a while)...")
perm_importance = permutation_importance(nn_best, X_test_scaled, y_test, 
                                        n_repeats=10, random_state=42, 
                                        scoring='roc_auc')

feature_importance_nn = pd.DataFrame({
    'Variable': X_train.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

print("\nNeural Network - Variable Importance (via Permutation):")
print(feature_importance_nn.head(10))

# Store results
results['NeuralNetwork'] = {
    'accuracy': accuracy_nn,
    'roc_auc': roc_auc_nn,
    'report': report_nn,
    'conf_matrix': conf_matrix_nn,
    'importance': feature_importance_nn,
    'model': nn_best,
    'y_pred': y_pred_nn,
    'y_prob': y_pred_prob_nn
}
# ----------------------------------------------
# Model Comparison and Visualization
# ----------------------------------------------
print("\n--- Model Comparison ---")

# Compare accuracies
accuracies = {
    'Logistic Regression': results['Logistic']['accuracy'],
    'Logistic Regression (Poly)': results['Logistic_Poly']['accuracy'],
    'Random Forest': results['RandomForest']['accuracy'],
    'Gradient Boosting': results['GradientBoosting']['accuracy'],
    'Neural Network': results['NeuralNetwork']['accuracy']
}
print("Model Accuracies:")
for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
    print(f"{model}: {acc:.4f}")

# Compare ROC AUC
roc_aucs = {
    'Logistic Regression': results['Logistic']['roc_auc'],
    'Logistic Regression (Poly)': results['Logistic_Poly']['roc_auc'],
    'Random Forest': results['RandomForest']['roc_auc'],
    'Gradient Boosting': results['GradientBoosting']['roc_auc'],
    'Neural Network': results['NeuralNetwork']['roc_auc']
}
print("\nModel ROC AUC Scores:")
for model, auc_score in sorted(roc_aucs.items(), key=lambda x: x[1], reverse=True):
    print(f"{model}: {auc_score:.4f}")

# ROC Curve Visualization
plt.figure(figsize=(12, 8))
plt.plot(fpr_logit, tpr_logit, label=f'Logistic (AUC = {roc_auc_logit:.3f})')
plt.plot(fpr_logit_poly, tpr_logit_poly, label=f'Logistic Poly (AUC = {roc_auc_logit_poly:.3f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_gb, tpr_gb, label=f'Gradient Boosting (AUC = {roc_auc_gb:.3f})')
plt.plot(fpr_nn, tpr_nn, label=f'Neural Network (AUC = {roc_auc_nn:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('roc_curves_comparison.png')
plt.close()

# ----------------------------------------------
# Feature Importance Comparison
# ----------------------------------------------
print("\n--- Creating Feature Importance Comparison ---")

# Get feature importances from all models
features = X_train.columns.tolist()

# Create a combined feature importance DataFrame for the non-polynomial models
combined_importance = pd.DataFrame({'Feature': features})

# Add importances from each model
for model_name in ['Logistic', 'RandomForest', 'GradientBoosting', 'NeuralNetwork']:
    model_imp = results[model_name]['importance']
    
    if model_name == 'Logistic':
        # For logistic regression, create a mapping from variable names to importance
        imp_dict = dict(zip(model_imp['Variable'], model_imp['Abs_Importance']))
        # Some variables might be interaction terms, so we'll get only the main effects
        combined_importance[f'{model_name}_Importance'] = [
            imp_dict.get(feature, 0) for feature in features
        ]
    else:
        # For tree models and neural network, create a mapping from variable names to importance
        imp_dict = dict(zip(model_imp['Variable'], model_imp['Importance']))
        combined_importance[f'{model_name}_Importance'] = [
            imp_dict.get(feature, 0) for feature in features
        ]

# Normalize importances for better comparison
for model_name in ['Logistic', 'RandomForest', 'GradientBoosting', 'NeuralNetwork']:
    col = f'{model_name}_Importance'
    if combined_importance[col].sum() > 0:  # Avoid division by zero
        combined_importance[col] = combined_importance[col] / combined_importance[col].sum()

# Sort by average importance
combined_importance['Avg_Importance'] = combined_importance[[
    'Logistic_Importance', 'RandomForest_Importance', 
    'GradientBoosting_Importance', 'NeuralNetwork_Importance'
]].mean(axis=1)
combined_importance = combined_importance.sort_values('Avg_Importance', ascending=False).reset_index(drop=True)

# Plot the top features
plt.figure(figsize=(12, 8))
top_n = 10
top_features = combined_importance.head(top_n)

x = np.arange(len(top_features))
width = 0.2  # Narrower bars to fit multiple models

plt.bar(x - 1.5*width, top_features['Logistic_Importance'], width, label='Logistic Regression')
plt.bar(x - 0.5*width, top_features['RandomForest_Importance'], width, label='Random Forest')
plt.bar(x + 0.5*width, top_features['GradientBoosting_Importance'], width, label='Gradient Boosting')
plt.bar(x + 1.5*width, top_features['NeuralNetwork_Importance'], width, label='Neural Network')

plt.xlabel('Features')
plt.ylabel('Normalized Importance')
plt.title('Feature Importance Comparison')
plt.xticks(x, top_features['Feature'], rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.close()

# ----------------------------------------------
# Polynomial Logistic Regression Analysis
# ----------------------------------------------
print("\n--- Polynomial Logistic Regression Analysis ---")

# Extract and display the top polynomial terms
poly_importance = results['Logistic_Poly']['importance']
print("Top 15 Important Terms in Polynomial Logistic Regression:")
print(poly_importance[['Variable', 'Coefficient']].head(15))

# Plot the top polynomial terms
plt.figure(figsize=(14, 8))
top_poly_terms = poly_importance.head(15)
colors = ['red' if coef < 0 else 'blue' for coef in top_poly_terms['Coefficient']]

plt.barh(top_poly_terms['Variable'], top_poly_terms['Abs_Importance'], color=colors)
plt.xlabel('Absolute Coefficient Magnitude')
plt.ylabel('Polynomial Terms')
plt.title('Top 15 Polynomial Terms in Logistic Regression')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('polynomial_terms_importance.png')
plt.close()

# Create a summary of the most important relationships
print("\nSummary of Important Relationships:")
print("Linear Terms:")
linear_terms = [term for term in poly_importance['Variable'] if '^' not in term and '*' not in term]
if linear_terms:
    linear_imp = poly_importance[poly_importance['Variable'].isin(linear_terms)].head(5)
    print(linear_imp[['Variable', 'Coefficient']])

print("\nQuadratic (Squared) Terms:")
squared_terms = [term for term in poly_importance['Variable'] if '^2' in term]
if squared_terms:
    squared_imp = poly_importance[poly_importance['Variable'].isin(squared_terms)].head(5)
    print(squared_imp[['Variable', 'Coefficient']])

print("\nInteraction Terms:")
interaction_terms = [term for term in poly_importance['Variable'] if '*' in term and '^' not in term]
if interaction_terms:
    interaction_imp = poly_importance[poly_importance['Variable'].isin(interaction_terms)].head(5)
    print(interaction_imp[['Variable', 'Coefficient']])

print("\nAnalysis completed! Results saved as images and model performance metrics displayed above.")

# ----------------------------------------------
# Partial Dependence Plots for Top Features
# ----------------------------------------------
print("\n--- Creating Partial Dependence Plots ---")

# Get top 3 features from each model
top_features_logit = results['Logistic']['importance']['Variable'].head(3).tolist()
top_features_rf = results['RandomForest']['importance']['Variable'].head(3).tolist()
top_features_gb = results['GradientBoosting']['importance']['Variable'].head(3).tolist()

# Combine and get unique features
top_features = list(set(top_features_logit + top_features_rf + top_features_gb))

# Function to create PDP plots - updated version without plot_partial_dependence
def create_pdp_plots(model, X_data, features, model_name):
    for feature in features:
        try:
            # Get feature index if working with column names
            if isinstance(feature, str):
                feature_idx = list(X_data.columns).index(feature)
            else:
                feature_idx = feature
            
            # Create the plot using the recommended API
            fig, ax = plt.subplots(figsize=(8, 6))
            display = PartialDependenceDisplay.from_estimator(
                model, X_data, [feature_idx], 
                kind="average", ax=ax, 
                line_kw={"color": "blue", "linewidth": 2},
                centered=True
            )
            ax.set_xlabel(feature if isinstance(feature, str) else X_data.columns[feature])
            ax.set_ylabel('Partial dependence')
            ax.set_title(f'Partial Dependence Plot for {feature} ({model_name})')
            ax.grid(True)
            plt.tight_layout()
            plt.savefig(f'pdp_{model_name}_{feature if isinstance(feature, str) else X_data.columns[feature]}.png')
            plt.close()
            
            print(f"Created PDP for {feature} ({model_name})")
        except Exception as e:
            print(f"Error creating PDP for {feature}: {e}")

# Create PDP plots for Random Forest and Gradient Boosting
create_pdp_plots(results['RandomForest']['model'], X_test_scaled_df, top_features, 'RandomForest')
create_pdp_plots(results['GradientBoosting']['model'], X_test_scaled_df, top_features, 'GradientBoosting')


# ----------------------------------------------
# Save trained models and scaler for later use
# ----------------------------------------------
print("\n--- Saving models and scaler ---")

import pickle
import os

# Create a directory to store models if it doesn't exist
model_dir = "saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Save the scaler
with open(os.path.join(model_dir, 'feature_scaler.pkl'), 'wb') as file:
    pickle.dump(scaler, file)
print("Feature scaler saved")

# List of models to save (update this based on what models you've trained)
model_files = {
    'LogisticRegression': 'logistic_model.pkl',
    'LogisticPoly': 'logistic_poly_model.pkl',
    'RandomForest': 'random_forest_model.pkl',
    'GradientBoosting': 'gradient_boosting_model.pkl',
    'NeuralNetwork': 'neural_network_model.pkl'
}

# Check what's in the results dictionary
print("\nAvailable models in results dictionary:")
for key in results.keys():
    print(f"  - {key}")

# Create a mapping between results keys and model file names
model_mapping = {
    'Logistic': 'LogisticRegression',
    'Logistic_Poly': 'LogisticPoly',
    'RandomForest': 'RandomForest',
    'GradientBoosting': 'GradientBoosting',
    'NeuralNetwork': 'NeuralNetwork',
    # Add other mappings if your keys are different
}

# Save each model, checking both the direct name and mapped name
for model_name, filename in model_files.items():
    # Check if model exists directly in results
    if model_name in results and 'model' in results[model_name]:
        with open(os.path.join(model_dir, filename), 'wb') as file:
            pickle.dump(results[model_name]['model'], file)
        print(f"{model_name} model saved")
    
    # Check if model exists via mapping
    elif model_name in model_mapping.values():
        # Find the key in the mapping that corresponds to this model name
        for key, value in model_mapping.items():
            if value == model_name and key in results and 'model' in results[key]:
                with open(os.path.join(model_dir, filename), 'wb') as file:
                    pickle.dump(results[key]['model'], file)
                print(f"{model_name} model saved (from {key})")
    else:
        print(f"⚠️ {model_name} not found in results dictionary")

# Also save the entire results dictionary for complete information
with open(os.path.join(model_dir, 'model_results.pkl'), 'wb') as file:
    pickle.dump(results, file)
print("Complete results dictionary saved")

# Save model feature names for reference
feature_names = X_train.columns.tolist()
with open(os.path.join(model_dir, 'feature_names.pkl'), 'wb') as file:
    pickle.dump(feature_names, file)
print("Feature names saved")

print(f"All models saved in '{model_dir}' directory")

# Print list of all saved files for verification
print("\nFiles saved in model directory:")
for file in os.listdir(model_dir):
    file_path = os.path.join(model_dir, file)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    print(f"  - {file} ({file_size:.2f} MB)")

import pandas as pd
import numpy as np
from tabulate import tabulate  # For pretty table output in terminal

# ---------------------------
# Extract & Prepare Feature Importance Data
# ---------------------------

# Logistic Regression Feature Importance
logit_importance = results["Logistic"]["importance"][["Variable", "Coefficient"]].copy()
logit_importance.rename(columns={"Variable": "Feature", "Coefficient": "Logit"}, inplace=True)

# Polynomial Logistic Regression
logit_poly_importance = results["Logistic_Poly"]["importance"][["Variable", "Coefficient"]].copy()
logit_poly_importance.rename(columns={"Variable": "Feature", "Coefficient": "Poly Logit"}, inplace=True)

# Random Forest Feature Importance
rf_importance_df = results["RandomForest"]["importance"][["Variable", "Importance"]].copy()
rf_importance_df.rename(columns={"Variable": "Feature", "Importance": "Random Forest"}, inplace=True)

# Gradient Boosting Feature Importance
gb_importance_df = results["GradientBoosting"]["importance"][["Variable", "Importance"]].copy()
gb_importance_df.rename(columns={"Variable": "Feature", "Importance": "Gradient Boosting"}, inplace=True)

# Neural Network Feature Importance (Permutation Importance)
nn_importance_df = results["NeuralNetwork"]["importance"][["Variable", "Importance"]].copy()
nn_importance_df.rename(columns={"Variable": "Feature", "Importance": "Neural Net"}, inplace=True)

# ---------------------------
# Combine All Feature Importances
# ---------------------------
feature_comparison = (
    logit_importance
    .merge(logit_poly_importance, on="Feature", how="outer")
    .merge(rf_importance_df, on="Feature", how="outer")
    .merge(gb_importance_df, on="Feature", how="outer")
    .merge(nn_importance_df, on="Feature", how="outer")
)

# Fill missing values with 0 for importance scores (no ranking if not selected)
importance_cols = ["Random Forest", "Gradient Boosting", "Neural Net"]
feature_comparison[importance_cols] = feature_comparison[importance_cols].fillna(0)

# Rank Features (Higher Importance = Lower Rank)
feature_comparison["Random Forest Rank"] = feature_comparison["Random Forest"].rank(ascending=False, method="min")
feature_comparison["Gradient Boosting Rank"] = feature_comparison["Gradient Boosting"].rank(ascending=False, method="min")
feature_comparison["Neural Net Rank"] = feature_comparison["Neural Net"].rank(ascending=False, method="min")

# Fill missing ranks with -1 (indicating "Not Ranked")
rank_columns = ["Random Forest Rank", "Gradient Boosting Rank", "Neural Net Rank"]
feature_comparison[rank_columns] = feature_comparison[rank_columns].fillna(-1).astype(int)

# ---------------------------
# Format the Table for Terminal Output
# ---------------------------

def format_logit(value):
    """Formats logistic regression coefficients with arrows to show direction."""
    if pd.isna(value): return "❌ Not significant"
    return f"⬆️ ({value:.2f})" if value > 0 else f"⬇️ ({value:.2f})"

def format_rank(value):
    """Formats rankings with medals for top 3 features."""
    if value == -1: return "❌ Not ranked"
    if value == 1: return "🏆 #1"
    if value == 2: return "🥈 #2"
    if value == 3: return "🥉 #3"
    return f"#{int(value)}"

# Create formatted table for terminal output
feature_comparison_terminal = feature_comparison.copy()
feature_comparison_terminal["Logit"] = feature_comparison_terminal["Logit"].apply(format_logit)
feature_comparison_terminal["Poly Logit"] = feature_comparison_terminal["Poly Logit"].apply(format_logit)
feature_comparison_terminal["Random Forest"] = feature_comparison_terminal["Random Forest Rank"].apply(format_rank)
feature_comparison_terminal["Gradient Boosting"] = feature_comparison_terminal["Gradient Boosting Rank"].apply(format_rank)
feature_comparison_terminal["Neural Net"] = feature_comparison_terminal["Neural Net Rank"].apply(format_rank)

# Drop intermediate ranking columns for terminal output
feature_comparison_terminal.drop(columns=["Random Forest Rank", "Gradient Boosting Rank", "Neural Net Rank"], inplace=True)

# ---------------------------
# Print & Save Results
# ---------------------------
print("\n=== FEATURE IMPORTANCE COMPARISON (Terminal) ===")
print(tabulate(feature_comparison_terminal, headers="keys", tablefmt="pretty", showindex=False))

# ---------------------------
# Save Clean CSV (No Icons)
# ---------------------------
# Remove icons for clean CSV output
def remove_icons(value):
    """Removes formatting icons from text output."""
    if isinstance(value, str):
        return value.replace("🏆", "").replace("🥈", "").replace("🥉", "").replace("⬆️", "").replace("⬇️", "").replace("❌", "").strip()
    return value

# Apply cleaning function
feature_comparison_csv = feature_comparison_terminal.applymap(remove_icons)

# Replace missing values with "Not Ranked" for better CSV readability
feature_comparison_csv = feature_comparison_csv.fillna("Not Ranked")

# Save to CSV
feature_comparison_csv.to_csv("feature_importance_comparison.csv", index=False)
print("\nFeature importance table saved as 'feature_importance_comparison.csv'.")

##############################################################
#file 3
#best map 2 final

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 18:03:11 2025

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:41:20 2025

@author: Ashley
"""
#file3

# -*- coding: utf-8 -*-
"""
Parallelized CartoPy + Folium Visualization of Viviparity Probability
- Reads real lat/lon points from CSV
- Dynamically calculates bounding box
- Efficiently extracts climate data from rasters using multiprocessing
- Predicts viviparity probability
- Generates interpolated contours for Folium map
"""

import os
import numpy as np
import pandas as pd
import rasterio
import pickle
import matplotlib.pyplot as plt
import geojsoncontour
import folium
import branca
from folium import plugins
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

###########################
# PATHS / MODEL
###########################
MODEL_DIR = "saved_models"
CLIMATE_DIR = "."
CSV_FILE = "combined_data_with_climate_and_elev_hand_mod.csv"

# Climate variables for raster extraction
CLIMATE_VARS = {
    'Elev': "wc2.1_30s_elev",
    'Tmax': "wc2.1_30s_tmax",
    'Tmin': "wc2.1_30s_tmin",
    'Tavg': "wc2.1_30s_tavg",
    'Wind': "wc2.1_30s_wind",
    'Prec': "wc2.1_30s_prec",
    'Vapr': "wc2.1_30s_vapr",
    'Srad': "wc2.1_30s_srad"
}

RESOLUTION = .05 # Changed from 5 to 0.5 as requested
DEBUG_MODE = False  # Set to True to print sample extracted values
MAX_WORKERS = 12  # Changed back to 12 threads for better performance
#relanttioship bewtween industrial waste/ air polution/ and radiation/ in predicting antibiotic reistances. 


###########################
# LOAD REAL DATA & BOUNDING BOX
###########################
def load_real_data(csv_file):
    """Load real latitude/longitude points and calculate bounding box."""
    df_real = pd.read_csv(csv_file)

    if not {'Latitude', 'Longitude'}.issubset(df_real.columns):
        raise ValueError("CSV file must contain 'Latitude' and 'Longitude' columns.")
        
    # Check if we have viviparity data (O/V or 0/1) in the file
    has_viviparity_data = False
    viviparity_col = None
    for col_name in ['Viviparity', 'viviparity', 'Viviparous', 'viviparous', 'ReproMode', 'Parity']:
        if col_name in df_real.columns:
            viviparity_col = col_name
            has_viviparity_data = True
            print(f"✅ Found viviparity data in column '{viviparity_col}'")
            # Print sample values to verify format
            sample_values = df_real[viviparity_col].dropna().unique()
            print(f"  - Sample values: {sample_values[:10]}")
            break
    
    if has_viviparity_data:
        # Identify the format - check if using O/V notation or 0/1 notation
        unique_values = set(str(x).upper() for x in df_real[viviparity_col].dropna().unique())
        print(f"  - Unique reproductive mode values: {unique_values}")
        
        # Create a new column for Viviparous (1) / Oviparous (0) classification
        df_real['IsViviparous'] = df_real[viviparity_col].apply(
            lambda x: 1 if str(x).upper() in ['1', 'TRUE', 'YES', 'V', 'VIVIPAROUS', 'VIVIPARITY'] else 0
        )
        
        v_count = df_real['IsViviparous'].sum()
        o_count = len(df_real) - v_count
        print(f"  - Viviparous (V/1): {v_count} species")
        print(f"  - Oviparous (O/0): {o_count} species")
        
        # Verify the conversion worked correctly
        if 'V' in unique_values or 'O' in unique_values:
            v_in_data = sum(1 for x in df_real[viviparity_col] if str(x).upper() == 'V')
            converted_v = sum(1 for i, x in enumerate(df_real[viviparity_col]) 
                             if str(x).upper() == 'V' and df_real['IsViviparous'].iloc[i] == 1)
            
            print(f"  - Verification: Found {v_in_data} 'V' values, converted {converted_v} to 1")
            
            if v_in_data != converted_v:
                print("⚠️ Warning: Conversion may not be accurate. Please check the data.")
    else:
        print("⚠️ No viviparity data found in CSV. Points will be colored uniformly.")
        df_real['IsViviparous'] = -1  # Unknown

    # Manually setting the bounding box for all of Europe and Asia
    # Manually setting the bounding box for all of Europe and Asia
    min_lat, max_lat = 36, 60  # Extended south from 45 to 40
    min_lon, max_lon = -30, 40  # Extended west from -25 to -30


    print(f"📌 Bounding Box: {min_lon}, {min_lat} to {max_lon}, {max_lat}")
    return df_real, min_lon, min_lat, max_lon, max_lat

###########################
# LOAD MODEL
###########################
def load_model(model_name='NeuralNetwork'):
    """Load model and related files from MODEL_DIR."""
    model_files = {
        'NeuralNetwork': 'neural_network_model.pkl',
        'RandomForest': 'random_forest_model.pkl'
    }
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        with open(os.path.join(MODEL_DIR, model_files[model_name]), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"Error: Could not find model files in {MODEL_DIR}: {e}")
        raise

###########################
# GENERATE GRID FOR MODEL
###########################
def make_lon_lat_grid(min_lon, min_lat, max_lon, max_lat, resolution=1.0):
    """Generate a grid of lat/lon points with equal spacing in actual distance."""
    # The issue with unequal spacing is because longitude degrees vary in physical distance
    # based on latitude (they get closer together as you move away from the equator)
    
    # Calculate approximate correction factor for longitude at the mean latitude
    # This ensures grid cells are roughly square in terms of actual distance on Earth
    mean_lat_radians = np.radians((min_lat + max_lat) / 2)
    lon_correction = np.cos(mean_lat_radians)
    
    # Account for the correction in longitude spacing
    lon_resolution = resolution / lon_correction
    
    print(f"Using resolution: {resolution}° latitude, {lon_resolution:.4f}° longitude")
    print(f"(Correction factor: {lon_correction:.4f} at latitude {(min_lat + max_lat)/2:.1f}°)")
    
    # Generate grid
    lons = np.arange(min_lon, max_lon, lon_resolution)
    lats = np.arange(min_lat, max_lat, resolution)
    
    # Create meshgrid with properly spaced points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Convert to DataFrame
    df_grid = pd.DataFrame({'Longitude': lon_grid.ravel(), 'Latitude': lat_grid.ravel()})
    
    print(f"✅ Created {len(df_grid)} grid points.")
    print(f"   Grid dimensions: {len(lats)} rows × {len(lons)} columns")
    print(f"   Latitude range: {min_lat} to {max_lat} ({len(lats)} points)")
    print(f"   Longitude range: {min_lon} to {max_lon} ({len(lons)} points)")
    
    return df_grid

###########################
# PARALLEL CLIMATE DATA EXTRACTION
###########################
def extract_climate_variable(df_grid, var_name, folder_name):
    """Efficiently extract climate data in batch mode, using multiple workers."""
    print(f"🔄 Extracting {var_name} for all points using {MAX_WORKERS} workers...")

    if var_name == "Elev":  # Special case for elevation (single file)
        file_path = os.path.join(CLIMATE_DIR, f"{folder_name}.tif")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist.")
            df_grid[var_name] = np.nan
            return df_grid
            
        values = process_raster_file(file_path, df_grid, var_name)
        df_grid[var_name] = values
        return df_grid

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for month in range(1, 13):
            file_path = os.path.join(CLIMATE_DIR, folder_name, f"{folder_name}_{month:02d}.tif")
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist.")
                continue
                
            futures.append(executor.submit(process_raster_file, file_path, df_grid, var_name))

        if not futures:
            print(f"Warning: No valid files found for {var_name}")
            df_grid[var_name] = np.nan
            return df_grid
            
        results = [f.result() for f in tqdm(futures, desc=f"Processing {var_name}")]

    if not results:
        df_grid[var_name] = np.nan
    else:
        df_grid[var_name] = np.nanmean(results, axis=0)  # Compute mean across 12 months
        
    print(f"✅ Extracted {var_name} for {len(df_grid)} points.")
    
    # Ensure we're always returning a DataFrame
    if not isinstance(df_grid, pd.DataFrame):
        print(f"Converting {var_name} result back to DataFrame")
        # This should never happen, but just in case
        temp_df = pd.DataFrame({'Longitude': df_grid['Longitude'], 'Latitude': df_grid['Latitude']})
        temp_df[var_name] = df_grid[var_name]
        df_grid = temp_df
        
    return df_grid

def process_raster_file(file_path, df_grid, var_name):
    """Read raster file and extract values for all points."""
    print(f"📂 Opening {file_path}...")
    values = np.full(len(df_grid), np.nan)

    try:
        with rasterio.open(file_path) as src:
            raster_data = src.read(1)
            nodata_value = src.nodata if src.nodata is not None else -3.4e+38

            for i, (lon, lat) in enumerate(zip(df_grid["Longitude"], df_grid["Latitude"])):
                try:
                    # Fixed from rasterio.index to src.index
                    row, col = src.index(lon, lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        pixel_value = raster_data[row, col]
                        # Improved NoData handling
                        if pixel_value == nodata_value or pixel_value < -100:
                            pixel_value = np.nan
                        values[i] = pixel_value

                        if DEBUG_MODE and i % 500 == 0:
                            print(f"📊 {var_name} at ({lon}, {lat}): {pixel_value}")
                except IndexError:
                    # Point outside raster bounds
                    if DEBUG_MODE and i % 500 == 0:
                        print(f"⚠️ Point ({lon}, {lat}) outside raster bounds")
                    values[i] = np.nan
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return values

    # Make sure we return just the values array, not a modified df_grid
    # This prevents the bug where df_grid becomes a NumPy array
    return values

###########################
# SAVE ENVIRONMENTAL DATA
###########################
def save_environmental_data(df_grid, output_file="environmental_values.csv"):
    """Save all environmental variables by latitude and longitude to a CSV file."""
    # Get all environmental variables (excluding Latitude, Longitude, and Probability)
    env_vars = [col for col in df_grid.columns if col not in ['Latitude', 'Longitude', 'Probability']]
    
    # Create a clean dataframe with lat, long, and all environmental variables
    df_env = df_grid[['Latitude', 'Longitude'] + env_vars].copy()
    
    # Calculate summary statistics
    summary = {}
    for var in env_vars:
        if df_env[var].notna().any():  # Only calculate if we have valid values
            summary[f"{var}_mean"] = df_env[var].mean()
            summary[f"{var}_median"] = df_env[var].median()
            summary[f"{var}_min"] = df_env[var].min()
            summary[f"{var}_max"] = df_env[var].max()
            summary[f"{var}_std"] = df_env[var].std()
    
    # Save data to CSV
    try:
        df_env.to_csv(output_file, index=False)
        print(f"✅ Environmental data saved to {output_file}")
        
        # Also save summary statistics
        summary_df = pd.DataFrame([summary])
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary statistics saved to {summary_file}")
        
        # Print summary to console
        print("\n📊 Summary of Environmental Variables:")
        for var in env_vars:
            if f"{var}_mean" in summary:
                print(f"{var}: Mean={summary[f'{var}_mean']:.2f}, Min={summary[f'{var}_min']:.2f}, Max={summary[f'{var}_max']:.2f}")
            else:
                print(f"{var}: No valid data")
        
    except Exception as e:
        print(f"❌ Error saving environmental data: {e}")
    
    return df_env

###########################
# PREDICT VIVIPARITY PROBABILITY
###########################
def predict_for_grid(df_grid, model, scaler, feature_list):
    """Extracts climate data -> predicts viviparity probability."""
    # Ensure df_grid is a DataFrame at the start
    if not isinstance(df_grid, pd.DataFrame):
        print("Warning: Input to predict_for_grid is not a DataFrame. Converting...")
        df_grid = pd.DataFrame(df_grid)
    
    for var in feature_list:
        if var not in CLIMATE_VARS:
            print(f"Warning: Climate variable '{var}' not found in CLIMATE_VARS dictionary. Skipping.")
            df_grid[var] = np.nan
            continue
            
        df_grid = extract_climate_variable(df_grid, var, CLIMATE_VARS[var])
        
        # Double-check that we still have a DataFrame after extraction
        if not isinstance(df_grid, pd.DataFrame):
            print(f"Warning: After extracting {var}, result is not a DataFrame. Converting back...")
            columns = list(CLIMATE_VARS.keys())
            columns = ['Latitude', 'Longitude'] + [c for c in columns if c in df_grid]
            temp_df = pd.DataFrame()
            for col in columns:
                if col in df_grid:
                    temp_df[col] = df_grid[col]
            df_grid = temp_df
    
    # Save environmental data to CSV before prediction
    save_environmental_data(df_grid)

    # Check if we have any valid data
    try:
        valid_rows = df_grid.dropna(subset=feature_list)
        if len(valid_rows) == 0:
            print("Warning: No valid data points after climate extraction. Check your raster files.")
            df_grid["Probability"] = np.nan
            return df_grid
    except AttributeError as e:
        print(f"Error: {e}")
        print(f"Type of df_grid: {type(df_grid)}")
        print("Converting to DataFrame and trying again...")
        # Try to recover by converting to DataFrame
        if hasattr(df_grid, 'keys'):
            df_grid = pd.DataFrame({k: df_grid[k] for k in df_grid.keys()})
        else:
            print("Cannot recover - df_grid doesn't have expected structure")
            raise

    X = scaler.transform(valid_rows[feature_list].to_numpy())
    y_prob = model.predict_proba(X)[:, 1]

    df_pred = df_grid.copy()
    df_pred["Probability"] = np.nan
    df_pred.loc[valid_rows.index, "Probability"] = y_prob
    
    print(f"✅ Prediction complete for {len(valid_rows)} points out of {len(df_grid)} total.")
    
    # Save final prediction data with environmental variables
    df_pred.to_csv("prediction_with_environmental_data.csv", index=False)
    print("✅ Complete prediction data saved to prediction_with_environmental_data.csv")
    
    return df_pred

###########################
# CREATE INTERPOLATED CONTOURS
###########################
def create_contours(df_pred):
    """Generate interpolated contours from predicted values with minimal smoothing to preserve detail."""
    # Filter out NaN values
    df_filtered = df_pred.dropna(subset=["Probability"])
    
    if len(df_filtered) < 10:
        print("Warning: Not enough valid points for interpolation.")
        # Return empty GeoJSON object (not string)
        return {"type": "FeatureCollection", "features": []}, None
        
    x, y, z = df_filtered["Longitude"], df_filtered["Latitude"], df_filtered["Probability"]

    # Create a higher resolution grid for more detailed interpolation
    grid_size = min(300, max(100, len(df_filtered) // 5))  # Increased resolution
    print(f"Using interpolation grid size: {grid_size}x{grid_size}")
    
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size), 
        np.linspace(y.min(), y.max(), grid_size)
    )
    
    # Use nearest interpolation for all points to avoid over-smoothing
    z_mesh = griddata((x, y), z, (x_mesh, y_mesh), method='nearest')
    
    # Only use cubic interpolation for visual smoothness, but preserve the detailed structure
    # by not replacing too many points
    z_cubic = griddata((x, y), z, (x_mesh, y_mesh), method='cubic')
    
    # Only replace nearest with cubic where cubic is valid and conditions are met
    mask = ~np.isnan(z_cubic)
    # Apply less blending to preserve the original data
    z_mesh[mask] = 0.9 * z_mesh[mask] + 0.1 * z_cubic[mask]
    
    # Apply very minimal smoothing to preserve details
    # Reduced sigma from [3,3] to [1,1] for much less smoothing
    z_mesh = ndimage.gaussian_filter(z_mesh, sigma=[.5, .5], mode='nearest')

    # Create more contour levels for finer detail
    levels = 15  # Increased from 10 to 15
    
    # Use a consistent colormap - "YlGnBu" for both the contours and legend
    cmap = plt.cm.YlGnBu
    
    # Create contours
    plt.figure(figsize=(1, 1))  # Small figure to minimize memory usage
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels=levels, cmap=cmap)
    try:
        geojson = geojsoncontour.contourf_to_geojson(contourf=contourf)
        # Ensure geojson is a dictionary, not a string
        if isinstance(geojson, str):
            print("Converting GeoJSON string to dictionary...")
            import json
            geojson = json.loads(geojson)
    except Exception as e:
        print(f"Error creating GeoJSON: {e}")
        geojson = {"type": "FeatureCollection", "features": []}
    
    plt.close()  # Close figure to free memory
    
    return geojson, cmap.name  # Return both the GeoJSON and the colormap name

###########################
# CREATE INTERACTIVE MAP (Folium)
###########################
def create_folium_map(df_pred, df_real, geojson_data):
    """Create an interactive Folium map overlaying contours and real data points."""
    # Unpack the geojson and colormap name
    if isinstance(geojson_data, tuple) and len(geojson_data) == 2:
        geojson, cmap_name = geojson_data
    else:
        geojson = geojson_data
        cmap_name = "YlGnBu"  # Default colormap
    
    # Use centroid for the initial map view
    center_lat = df_real['Latitude'].mean()
    center_lon = df_real['Longitude'].mean()
    
    # Fallback to hardcoded values if needed
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = 45, 10
    
    folium_map = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=4,  # Wider initial view
        tiles="cartodbpositron"
    )

    # Add contour overlay if available
    # Make sure geojson is a dictionary, not a string
    if isinstance(geojson, str):
        try:
            import json
            geojson = json.loads(geojson)
        except:
            print("Warning: Could not parse GeoJSON string")
            geojson = {"type": "FeatureCollection", "features": []}
    
    # Check if geojson is a dictionary with features
    if isinstance(geojson, dict) and "features" in geojson and geojson["features"]:
        folium.GeoJson(
            geojson, 
            style_function=lambda x: {
                'fillColor': x['properties']['fill'], 
                'opacity': 0.7,  # Slightly increased opacity
                'fillOpacity': 0.7,  # Increased opacity to make details more visible
                'weight': 0.5  # Thinner lines between contours for less visual interference
            }
        ).add_to(folium_map)
        
        # Add colorbar legend that matches the colormap used in the contours
        # Define color scales based on the colormap name
        color_scales = {
            "YlGnBu": ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'],
            "Blues": ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            "BuPu": ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
            "Greens": ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
            "Reds": ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
        }
        
        # Use the appropriate color scale, defaulting to YlGnBu if not found
        colors = color_scales.get(cmap_name, color_scales["YlGnBu"])
        
        colormap = branca.colormap.LinearColormap(
            colors=colors, 
            index=np.linspace(0, 1, len(colors)),
            vmin=0,
            vmax=1,
            caption='Probability of Viviparity'
        )
        folium_map.add_child(colormap)
    else:
        print("Warning: No valid GeoJSON features found for contour overlay")
        
    # Create a separate layer for points showing exact probability values
    points_layer = folium.FeatureGroup(name="Sample Points (toggle on/off)")
    
    # Add a subset of points with probability values (to avoid cluttering the map)
    # Use systematic sampling to get a representative distribution
    sample_size = min(500, len(df_pred))  # Limit to 500 points max
    step = max(1, len(df_pred) // sample_size)
    
    df_sample = df_pred.iloc[::step].dropna(subset=["Probability"])
    
    for _, point in df_sample.iterrows():
        # Skip points without valid probability
        if np.isnan(point.get("Probability", np.nan)):
            continue
            
        # Create color based on probability
        prob_color = plt.cm.YlGnBu(point["Probability"])
        # Convert RGBA to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(prob_color[0]*255), 
            int(prob_color[1]*255), 
            int(prob_color[2]*255)
        )
        
        # Create popup with detailed information
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Grid Point</b><br>
            Lat: {point.Latitude:.4f}<br>
            Lon: {point.Longitude:.4f}<br>
            <b>Probability:</b> {point.Probability:.3f}<br>
            <hr style="margin: 5px 0;">
            <b>Environmental Data:</b><br>
        """
        
        # Add environmental variables
        for var in [col for col in point.index if col not in ['Latitude', 'Longitude', 'Probability', 'distance']]:
            if not pd.isna(point[var]):
                popup_html += f"{var}: {point[var]:.1f}<br>"
        
        popup_html += "</div>"
        
        # Add circle marker
        folium.CircleMarker(
            location=[point.Latitude, point.Longitude],
            radius=2,  # Small points
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(points_layer)
    
    # Add the points layer to the map but set it to off by default
    points_layer.add_to(folium_map)
    
    # Create separate layers for viviparous and oviparous species
    viviparous_layer = folium.FeatureGroup(name="Viviparous Species (1)", show=True)
    oviparous_layer = folium.FeatureGroup(name="Oviparous Species (0)", show=True)
    unknown_layer = folium.FeatureGroup(name="Unknown Reproductive Mode", show=True)
    
    # Add real data points, colored by their reproductive mode
    for _, row in df_real.iterrows():
        if np.isnan(row.Latitude) or np.isnan(row.Longitude):
            continue
            
        # Build popup content
        popup_text = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Observation Point</b><br>
            Lat: {row.Latitude:.4f}<br>
            Lon: {row.Longitude:.4f}<br>
        """
        
        if 'Species' in row:
            popup_text += f"<b>Species:</b> {row.Species}<br>"
        
        if 'IsViviparous' in row:
            if row.IsViviparous == 1:
                popup_text += "<b>Reproductive Mode:</b> Viviparous<br>"
            elif row.IsViviparous == 0:
                popup_text += "<b>Reproductive Mode:</b> Oviparous<br>"
            else:
                popup_text += "<b>Reproductive Mode:</b> Unknown<br>"
                
        popup_text += "</div>"
                
        # Create marker with appropriate color
        if 'IsViviparous' in row and row.IsViviparous == 1:
            # Viviparous - use darker blue
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.9
            ).add_to(viviparous_layer)
        elif 'IsViviparous' in row and row.IsViviparous == 0:
            # Oviparous - use lighter yellow/orange
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.9
            ).add_to(oviparous_layer)
        else:
            # Unknown - use red or gray
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.9
            ).add_to(unknown_layer)
    
    # Add all layers to the map
    viviparous_layer.add_to(folium_map)
    oviparous_layer.add_to(folium_map)
    unknown_layer.add_to(folium_map)
    
    # Add a layer for predicted vs actual match assessment
    if 'IsViviparous' in df_real.columns and df_real['IsViviparous'].isin([0, 1]).any():
        assessment_layer = folium.FeatureGroup(name="Prediction Assessment", show=False)
        
        # For each observation point with known reproductive mode
        for _, row in df_real[df_real['IsViviparous'].isin([0, 1])].iterrows():
            # Find nearest prediction point
            df_pred['temp_dist'] = np.sqrt(
                (df_pred['Latitude'] - row.Latitude)**2 + 
                (df_pred['Longitude'] - row.Longitude)**2
            )
            nearest_pred = df_pred.loc[df_pred['temp_dist'].idxmin()]
            
            # Skip if no valid prediction
            if pd.isna(nearest_pred.get('Probability', np.nan)):
                continue
                
            # Calculate match percentage
            if row.IsViviparous == 1:
                match_pct = nearest_pred.Probability * 100
                correct_pred = nearest_pred.Probability >= 0.5
            else:  # IsViviparous == 0
                match_pct = (1 - nearest_pred.Probability) * 100
                correct_pred = nearest_pred.Probability < 0.5
            
            # Determine color based on match quality
            if correct_pred:
                # Good prediction - use green with intensity based on confidence
                color = f'#{int(155 + match_pct):02x}ff{int(155):02x}'
            else:
                # Poor prediction - use red with intensity based on error
                color = f'#ff{int(155 + (100-match_pct)):02x}{int(155):02x}'
            
            # Add marker showing prediction quality
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Prediction Assessment</b><br>
                Actual: {"Viviparous" if row.IsViviparous == 1 else "Oviparous"}<br>
                Predicted Probability: {nearest_pred.Probability:.3f}<br>
                <b>Match: {match_pct:.1f}%</b><br>
                <b>Outcome: {"✓ Correct" if correct_pred else "✗ Incorrect"}</b>
            </div>
            """
            
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            ).add_to(assessment_layer)
        
        # Clean up temporary column
        if 'temp_dist' in df_pred.columns:
            df_pred.drop('temp_dist', axis=1, inplace=True)
            
        # Add the assessment layer to the map
        assessment_layer.add_to(folium_map)
        
        # Add a legend explaining the colors
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:12px;
                    background-color: white; padding: 10px;
                    border-radius: 5px;">
            <span style="color: blue;"><b>●</b></span> Viviparous Species (1)<br>
            <span style="color: orange;"><b>●</b></span> Oviparous Species (0)<br>
            <span style="color: red;"><b>●</b></span> Unknown Reproductive Mode<br>
            <hr style="margin: 5px 0;">
            <i>Toggle layers using the control panel</i>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(legend_html))

    # Add map controls
    folium.LayerControl(collapsed=False).add_to(folium_map)
    plugins.Fullscreen().add_to(folium_map)
    plugins.MeasureControl().add_to(folium_map)
    
    # Save the map
    try:
        folium_map.save("viviparity_map_nn.html")
        print("✅ Map saved as 'viviparity_map.html'")
        
        # Also save a more detailed version
        folium_map.save("viviparity_map_detailed_nn.html")
        print("✅ Detailed map saved as 'viviparity_map_detailed.html'")
    except Exception as e:
        print(f"Error saving map: {e}")

###########################
# ANALYZE ENVIRONMENTAL DATA
###########################
def analyze_environmental_data(df_grid, df_real=None):
    """Perform additional analysis on environmental data."""
    # Get all environmental variables
    env_vars = [col for col in df_grid.columns if col not in ['Latitude', 'Longitude', 'Probability']]
    
    if len(env_vars) == 0:
        print("⚠️ No environmental variables found for analysis")
        return
    
    print("\n📊 Analyzing environmental data patterns...")
    
    # Create a grid for visualization
    try:
        # For each environmental variable, calculate statistics by latitude band
        lat_bands = pd.cut(df_grid['Latitude'], bins=10)
        lat_analysis = df_grid.groupby(lat_bands, observed=False)[env_vars].mean()
        lat_analysis.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in lat_analysis.index]
        
        # Save latitude band analysis
        lat_analysis.to_csv("environmental_by_latitude.csv")
        print("✅ Latitude band analysis saved to environmental_by_latitude.csv")
        
        # For each environmental variable, calculate statistics by longitude band
        lon_bands = pd.cut(df_grid['Longitude'], bins=10)
        lon_analysis = df_grid.groupby(lon_bands, observed=False)[env_vars].mean()
        lon_analysis.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in lon_analysis.index]
        
        # Save longitude band analysis
        lon_analysis.to_csv("environmental_by_longitude.csv")
        print("✅ Longitude band analysis saved to environmental_by_longitude.csv")
        
        # If we have real data points, compare environmental conditions at those points
        if df_real is not None and len(df_real) > 0:
            # For each real data point, extract nearest grid point's environmental data
            real_env_data = []
            
            for _, real_row in df_real.iterrows():
                # Calculate distance to each grid point
                df_grid['distance'] = np.sqrt(
                    (df_grid['Latitude'] - real_row['Latitude'])**2 + 
                    (df_grid['Longitude'] - real_row['Longitude'])**2
                )
                
                # Get closest point
                closest_idx = df_grid['distance'].idxmin()
                closest_point = df_grid.loc[closest_idx].copy()
                
                # Add real point info
                if 'Species' in real_row:
                    closest_point['Species'] = real_row['Species']
                
                # Add to collection
                real_env_data.append(closest_point)
            
            # Create DataFrame with environmental data at real points
            df_real_env = pd.DataFrame(real_env_data)
            
            # Save to CSV
            df_real_env.to_csv("environmental_at_real_points.csv", index=False)
            print("✅ Environmental data at real points saved to environmental_at_real_points.csv")
    
    except Exception as e:
        print(f"⚠️ Error in environmental analysis: {e}")

###########################
# MAIN
###########################
def main():
    try:
        print("Starting viviparity probability mapping...")
        
        # Load real data points
        df_real, min_lon, min_lat, max_lon, max_lat = load_real_data(CSV_FILE)
        
        # Load model
        model, scaler, features = load_model()
        
        # Create grid of points
        df_grid = make_lon_lat_grid(min_lon, min_lat, max_lon, max_lat, RESOLUTION)
        
        # Predict viviparity probability
        df_pred = predict_for_grid(df_grid, model, scaler, features)
        
        # Perform additional environmental analysis
        analyze_environmental_data(df_pred, df_real)
        
        # Create contours
        geojson_data = create_contours(df_pred)
        
        # Create interactive map
        create_folium_map(df_pred, df_real, geojson_data)
        
        print("\n✅ Process completed successfully!")
        print("📄 Output files:")
        print("  - environmental_values.csv (All environmental data by lat/long)")
        print("  - environmental_values_summary.csv (Summary statistics)")
        print("  - prediction_with_environmental_data.csv (Complete dataset with predictions)")
        print("  - environmental_by_latitude.csv (Environmental trends by latitude)")
        print("  - environmental_by_longitude.csv (Environmental trends by longitude)")
        print("  - environmental_at_real_points.csv (Environmental data at observation points)")
        print("  - viviparity_map.html (Interactive visualization)")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#############################################################
#file 4
#log model map

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:07:28 2025

@author: Ashley
"""
#file4

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 18:03:11 2025

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:41:20 2025

@author: Ashley
"""

# -*- coding: utf-8 -*-
"""
Parallelized CartoPy + Folium Visualization of Viviparity Probability
- Reads real lat/lon points from CSV
- Dynamically calculates bounding box
- Efficiently extracts climate data from rasters using multiprocessing
- Predicts viviparity probability
- Generates interpolated contours for Folium map
"""

import os
import numpy as np
import pandas as pd
import rasterio
import pickle
import matplotlib.pyplot as plt
import geojsoncontour
import folium
import branca
from folium import plugins
from scipy.interpolate import griddata
import scipy.ndimage as ndimage
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

###########################
# PATHS / MODEL
###########################
MODEL_DIR = "saved_models"
CLIMATE_DIR = "."
CSV_FILE = "combined_data_with_climate_and_elev_hand_mod.csv"

# Climate variables for raster extraction
CLIMATE_VARS = {
    'Elev': "wc2.1_30s_elev",
    'Tmax': "wc2.1_30s_tmax",
    'Tmin': "wc2.1_30s_tmin",
    'Tavg': "wc2.1_30s_tavg",
    'Wind': "wc2.1_30s_wind",
    'Prec': "wc2.1_30s_prec",
    'Vapr': "wc2.1_30s_vapr",
    'Srad': "wc2.1_30s_srad"
}

RESOLUTION = .05 # Changed from 5 to 0.5 as requested
DEBUG_MODE = False  # Set to True to print sample extracted values
MAX_WORKERS = 12  # Changed back to 12 threads for better performance
#relanttioship bewtween industrial waste/ air polution/ and radiation/ in predicting antibiotic reistances. 


###########################
# LOAD REAL DATA & BOUNDING BOX
###########################
def load_real_data(csv_file):
    """Load real latitude/longitude points and calculate bounding box."""
    df_real = pd.read_csv(csv_file)

    if not {'Latitude', 'Longitude'}.issubset(df_real.columns):
        raise ValueError("CSV file must contain 'Latitude' and 'Longitude' columns.")
        
    # Check if we have viviparity data (O/V or 0/1) in the file
    has_viviparity_data = False
    viviparity_col = None
    for col_name in ['Viviparity', 'viviparity', 'Viviparous', 'viviparous', 'ReproMode', 'Parity']:
        if col_name in df_real.columns:
            viviparity_col = col_name
            has_viviparity_data = True
            print(f"✅ Found viviparity data in column '{viviparity_col}'")
            # Print sample values to verify format
            sample_values = df_real[viviparity_col].dropna().unique()
            print(f"  - Sample values: {sample_values[:10]}")
            break
    
    if has_viviparity_data:
        # Identify the format - check if using O/V notation or 0/1 notation
        unique_values = set(str(x).upper() for x in df_real[viviparity_col].dropna().unique())
        print(f"  - Unique reproductive mode values: {unique_values}")
        
        # Create a new column for Viviparous (1) / Oviparous (0) classification
        df_real['IsViviparous'] = df_real[viviparity_col].apply(
            lambda x: 1 if str(x).upper() in ['1', 'TRUE', 'YES', 'V', 'VIVIPAROUS', 'VIVIPARITY'] else 0
        )
        
        v_count = df_real['IsViviparous'].sum()
        o_count = len(df_real) - v_count
        print(f"  - Viviparous (V/1): {v_count} species")
        print(f"  - Oviparous (O/0): {o_count} species")
        
        # Verify the conversion worked correctly
        if 'V' in unique_values or 'O' in unique_values:
            v_in_data = sum(1 for x in df_real[viviparity_col] if str(x).upper() == 'V')
            converted_v = sum(1 for i, x in enumerate(df_real[viviparity_col]) 
                             if str(x).upper() == 'V' and df_real['IsViviparous'].iloc[i] == 1)
            
            print(f"  - Verification: Found {v_in_data} 'V' values, converted {converted_v} to 1")
            
            if v_in_data != converted_v:
                print("⚠️ Warning: Conversion may not be accurate. Please check the data.")
    else:
        print("⚠️ No viviparity data found in CSV. Points will be colored uniformly.")
        df_real['IsViviparous'] = -1  # Unknown

    # Manually setting the bounding box for all of Europe and Asia
    # Manually setting the bounding box for all of Europe and Asia
    min_lat, max_lat = 36, 60  # Extended south from 45 to 40
    min_lon, max_lon = -30, 40  # Extended west from -25 to -30


    print(f"📌 Bounding Box: {min_lon}, {min_lat} to {max_lon}, {max_lat}")
    return df_real, min_lon, min_lat, max_lon, max_lat

def load_model(model_name='LogisticRegression'):
    """Load model and related files from MODEL_DIR."""
    model_files = {
        'NeuralNetwork': 'neural_network_model.pkl',
        'RandomForest': 'random_forest_model.pkl',
        'LogisticRegression': 'logistic_model.pkl',  # Add this line
        'LogisticPoly': 'logistic_poly_model.pkl'  # In case you want to use the polynomial version
    }
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    try:
        with open(os.path.join(MODEL_DIR, model_files[model_name]), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, feature_names
    except FileNotFoundError as e:
        print(f"Error: Could not find model files in {MODEL_DIR}: {e}")
        raise
###########################
# GENERATE GRID FOR MODEL
###########################
def make_lon_lat_grid(min_lon, min_lat, max_lon, max_lat, resolution=1.0):
    """Generate a grid of lat/lon points with equal spacing in actual distance."""
    # The issue with unequal spacing is because longitude degrees vary in physical distance
    # based on latitude (they get closer together as you move away from the equator)
    
    # Calculate approximate correction factor for longitude at the mean latitude
    # This ensures grid cells are roughly square in terms of actual distance on Earth
    mean_lat_radians = np.radians((min_lat + max_lat) / 2)
    lon_correction = np.cos(mean_lat_radians)
    
    # Account for the correction in longitude spacing
    lon_resolution = resolution / lon_correction
    
    print(f"Using resolution: {resolution}° latitude, {lon_resolution:.4f}° longitude")
    print(f"(Correction factor: {lon_correction:.4f} at latitude {(min_lat + max_lat)/2:.1f}°)")
    
    # Generate grid
    lons = np.arange(min_lon, max_lon, lon_resolution)
    lats = np.arange(min_lat, max_lat, resolution)
    
    # Create meshgrid with properly spaced points
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Convert to DataFrame
    df_grid = pd.DataFrame({'Longitude': lon_grid.ravel(), 'Latitude': lat_grid.ravel()})
    
    print(f"✅ Created {len(df_grid)} grid points.")
    print(f"   Grid dimensions: {len(lats)} rows × {len(lons)} columns")
    print(f"   Latitude range: {min_lat} to {max_lat} ({len(lats)} points)")
    print(f"   Longitude range: {min_lon} to {max_lon} ({len(lons)} points)")
    
    return df_grid

###########################
# PARALLEL CLIMATE DATA EXTRACTION
###########################
def extract_climate_variable(df_grid, var_name, folder_name):
    """Efficiently extract climate data in batch mode, using multiple workers."""
    print(f"🔄 Extracting {var_name} for all points using {MAX_WORKERS} workers...")

    if var_name == "Elev":  # Special case for elevation (single file)
        file_path = os.path.join(CLIMATE_DIR, f"{folder_name}.tif")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist.")
            df_grid[var_name] = np.nan
            return df_grid
            
        values = process_raster_file(file_path, df_grid, var_name)
        df_grid[var_name] = values
        return df_grid

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for month in range(1, 13):
            file_path = os.path.join(CLIMATE_DIR, folder_name, f"{folder_name}_{month:02d}.tif")
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist.")
                continue
                
            futures.append(executor.submit(process_raster_file, file_path, df_grid, var_name))

        if not futures:
            print(f"Warning: No valid files found for {var_name}")
            df_grid[var_name] = np.nan
            return df_grid
            
        results = [f.result() for f in tqdm(futures, desc=f"Processing {var_name}")]

    if not results:
        df_grid[var_name] = np.nan
    else:
        df_grid[var_name] = np.nanmean(results, axis=0)  # Compute mean across 12 months
        
    print(f"✅ Extracted {var_name} for {len(df_grid)} points.")
    
    # Ensure we're always returning a DataFrame
    if not isinstance(df_grid, pd.DataFrame):
        print(f"Converting {var_name} result back to DataFrame")
        # This should never happen, but just in case
        temp_df = pd.DataFrame({'Longitude': df_grid['Longitude'], 'Latitude': df_grid['Latitude']})
        temp_df[var_name] = df_grid[var_name]
        df_grid = temp_df
        
    return df_grid

def process_raster_file(file_path, df_grid, var_name):
    """Read raster file and extract values for all points."""
    print(f"📂 Opening {file_path}...")
    values = np.full(len(df_grid), np.nan)

    try:
        with rasterio.open(file_path) as src:
            raster_data = src.read(1)
            nodata_value = src.nodata if src.nodata is not None else -3.4e+38

            for i, (lon, lat) in enumerate(zip(df_grid["Longitude"], df_grid["Latitude"])):
                try:
                    # Fixed from rasterio.index to src.index
                    row, col = src.index(lon, lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        pixel_value = raster_data[row, col]
                        # Improved NoData handling
                        if pixel_value == nodata_value or pixel_value < -100:
                            pixel_value = np.nan
                        values[i] = pixel_value

                        if DEBUG_MODE and i % 500 == 0:
                            print(f"📊 {var_name} at ({lon}, {lat}): {pixel_value}")
                except IndexError:
                    # Point outside raster bounds
                    if DEBUG_MODE and i % 500 == 0:
                        print(f"⚠️ Point ({lon}, {lat}) outside raster bounds")
                    values[i] = np.nan
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return values

    # Make sure we return just the values array, not a modified df_grid
    # This prevents the bug where df_grid becomes a NumPy array
    return values

###########################
# SAVE ENVIRONMENTAL DATA
###########################
def save_environmental_data(df_grid, output_file="environmental_values.csv"):
    """Save all environmental variables by latitude and longitude to a CSV file."""
    # Get all environmental variables (excluding Latitude, Longitude, and Probability)
    env_vars = [col for col in df_grid.columns if col not in ['Latitude', 'Longitude', 'Probability']]
    
    # Create a clean dataframe with lat, long, and all environmental variables
    df_env = df_grid[['Latitude', 'Longitude'] + env_vars].copy()
    
    # Calculate summary statistics
    summary = {}
    for var in env_vars:
        if df_env[var].notna().any():  # Only calculate if we have valid values
            summary[f"{var}_mean"] = df_env[var].mean()
            summary[f"{var}_median"] = df_env[var].median()
            summary[f"{var}_min"] = df_env[var].min()
            summary[f"{var}_max"] = df_env[var].max()
            summary[f"{var}_std"] = df_env[var].std()
    
    # Save data to CSV
    try:
        df_env.to_csv(output_file, index=False)
        print(f"✅ Environmental data saved to {output_file}")
        
        # Also save summary statistics
        summary_df = pd.DataFrame([summary])
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary statistics saved to {summary_file}")
        
        # Print summary to console
        print("\n📊 Summary of Environmental Variables:")
        for var in env_vars:
            if f"{var}_mean" in summary:
                print(f"{var}: Mean={summary[f'{var}_mean']:.2f}, Min={summary[f'{var}_min']:.2f}, Max={summary[f'{var}_max']:.2f}")
            else:
                print(f"{var}: No valid data")
        
    except Exception as e:
        print(f"❌ Error saving environmental data: {e}")
    
    return df_env

def predict_for_grid(df_grid, model, scaler, feature_list):
    """
    Extracts climate data -> predicts viviparity probability 
    for the specific logistic regression model with Wind:Elev interaction
    """
    import statsmodels.api as sm
    import patsy

    # Ensure df_grid is a DataFrame at the start
    if not isinstance(df_grid, pd.DataFrame):
        print("Warning: Input to predict_for_grid is not a DataFrame. Converting...")
        df_grid = pd.DataFrame(df_grid)
    
    # Extract climate variables
    main_features = ['Tmax', 'Tmin', 'Tavg', 'Wind', 'Prec', 'Vapr', 'Srad', 'Elev']
    
    for var in main_features:
        if var not in CLIMATE_VARS:
            print(f"Warning: Climate variable '{var}' not found in CLIMATE_VARS dictionary. Skipping.")
            df_grid[var] = np.nan
            continue
            
        df_grid = extract_climate_variable(df_grid, var, CLIMATE_VARS[var])
    
    # Save environmental data to CSV before prediction
    save_environmental_data(df_grid)

    # Check if we have any valid data
    valid_rows = df_grid.dropna(subset=main_features).copy()
    if len(valid_rows) == 0:
        print("Warning: No valid data points after climate extraction. Check your raster files.")
        df_grid["Probability"] = np.nan
        return df_grid

    # Create specific interaction term (Wind:Elev)
    valid_rows['Wind:Elev'] = valid_rows['Wind'] * valid_rows['Elev']

    # Prepare the design matrix for prediction
    X_pred_df = valid_rows[main_features]
    
    # Scale the main features
    X_pred_scaled = pd.DataFrame(
        scaler.transform(X_pred_df), 
        columns=main_features, 
        index=X_pred_df.index
    )
    
    # Add the interaction term AFTER scaling
    X_pred_scaled['Wind:Elev'] = valid_rows['Wind'] * valid_rows['Elev']
    
    # Predict probabilities
    y_prob = model.predict(X_pred_scaled)
    
    # Create a copy of the original dataframe to preserve all columns
    df_pred = df_grid.copy()
    df_pred["Probability"] = np.nan
    df_pred.loc[valid_rows.index, "Probability"] = y_prob
    
    print(f"✅ Prediction complete for {len(valid_rows)} points out of {len(df_grid)} total.")
    
    # Save final prediction data with environmental variables
    df_pred.to_csv("prediction_with_environmental_data_logit.csv", index=False)
    print("✅ Complete prediction data saved to prediction_with_environmental_data_logit.csv")
    
    return df_pred

def _create_interaction_matrix(X, feature_names):
    """
    Create a design matrix with all main effects and interaction terms
    
    Parameters:
    X (numpy.ndarray): Scaled feature matrix
    feature_names (list): Names of original features
    
    Returns:
    numpy.ndarray: Design matrix with main effects and interactions
    """
    # First, add a column of 1s for the intercept
    interactions = [np.ones(X.shape[0])]
    
    # Add main effects
    interactions.extend([X[:, i] for i in range(X.shape[1])])
    
    # Add interaction terms
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            interactions.append(X[:, i] * X[:, j])
    
    return np.column_stack(interactions)

###########################
# CREATE INTERPOLATED CONTOURS
###########################
def create_contours(df_pred):
    """Generate interpolated contours from predicted values with minimal smoothing to preserve detail."""
    # Filter out NaN values
    df_filtered = df_pred.dropna(subset=["Probability"])
    
    if len(df_filtered) < 10:
        print("Warning: Not enough valid points for interpolation.")
        # Return empty GeoJSON object (not string)
        return {"type": "FeatureCollection", "features": []}, None
        
    x, y, z = df_filtered["Longitude"], df_filtered["Latitude"], df_filtered["Probability"]

    # Create a higher resolution grid for more detailed interpolation
    grid_size = min(300, max(100, len(df_filtered) // 5))  # Increased resolution
    print(f"Using interpolation grid size: {grid_size}x{grid_size}")
    
    x_mesh, y_mesh = np.meshgrid(
        np.linspace(x.min(), x.max(), grid_size), 
        np.linspace(y.min(), y.max(), grid_size)
    )
    
    # Use nearest interpolation for all points to avoid over-smoothing
    z_mesh = griddata((x, y), z, (x_mesh, y_mesh), method='nearest')
    
    # Only use cubic interpolation for visual smoothness, but preserve the detailed structure
    # by not replacing too many points
    z_cubic = griddata((x, y), z, (x_mesh, y_mesh), method='cubic')
    
    # Only replace nearest with cubic where cubic is valid and conditions are met
    mask = ~np.isnan(z_cubic)
    # Apply less blending to preserve the original data
    z_mesh[mask] = 0.9 * z_mesh[mask] + 0.1 * z_cubic[mask]
    
    # Apply very minimal smoothing to preserve details
    # Reduced sigma from [3,3] to [1,1] for much less smoothing
    z_mesh = ndimage.gaussian_filter(z_mesh, sigma=[.5, .5], mode='nearest')

    # Create more contour levels for finer detail
    levels = 15  # Increased from 10 to 15
    
    # Use a consistent colormap - "YlGnBu" for both the contours and legend
    cmap = plt.cm.YlGnBu
    
    # Create contours
    plt.figure(figsize=(1, 1))  # Small figure to minimize memory usage
    contourf = plt.contourf(x_mesh, y_mesh, z_mesh, levels=levels, cmap=cmap)
    try:
        geojson = geojsoncontour.contourf_to_geojson(contourf=contourf)
        # Ensure geojson is a dictionary, not a string
        if isinstance(geojson, str):
            print("Converting GeoJSON string to dictionary...")
            import json
            geojson = json.loads(geojson)
    except Exception as e:
        print(f"Error creating GeoJSON: {e}")
        geojson = {"type": "FeatureCollection", "features": []}
    
    plt.close()  # Close figure to free memory
    
    return geojson, cmap.name  # Return both the GeoJSON and the colormap name

###########################
# CREATE INTERACTIVE MAP (Folium)
###########################
def create_folium_map(df_pred, df_real, geojson_data):
    """Create an interactive Folium map overlaying contours and real data points."""
    # Unpack the geojson and colormap name
    if isinstance(geojson_data, tuple) and len(geojson_data) == 2:
        geojson, cmap_name = geojson_data
    else:
        geojson = geojson_data
        cmap_name = "YlGnBu"  # Default colormap
    
    # Use centroid for the initial map view
    center_lat = df_real['Latitude'].mean()
    center_lon = df_real['Longitude'].mean()
    
    # Fallback to hardcoded values if needed
    if np.isnan(center_lat) or np.isnan(center_lon):
        center_lat, center_lon = 45, 10
    
    folium_map = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=4,  # Wider initial view
        tiles="cartodbpositron"
    )

    # Add contour overlay if available
    # Make sure geojson is a dictionary, not a string
    if isinstance(geojson, str):
        try:
            import json
            geojson = json.loads(geojson)
        except:
            print("Warning: Could not parse GeoJSON string")
            geojson = {"type": "FeatureCollection", "features": []}
    
    # Check if geojson is a dictionary with features
    if isinstance(geojson, dict) and "features" in geojson and geojson["features"]:
        folium.GeoJson(
            geojson, 
            style_function=lambda x: {
                'fillColor': x['properties']['fill'], 
                'opacity': 0.7,  # Slightly increased opacity
                'fillOpacity': 0.7,  # Increased opacity to make details more visible
                'weight': 0.5  # Thinner lines between contours for less visual interference
            }
        ).add_to(folium_map)
        
        # Add colorbar legend that matches the colormap used in the contours
        # Define color scales based on the colormap name
        color_scales = {
            "YlGnBu": ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58'],
            "Blues": ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
            "BuPu": ['#f7fcfd', '#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
            "Greens": ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
            "Reds": ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
        }
        
        # Use the appropriate color scale, defaulting to YlGnBu if not found
        colors = color_scales.get(cmap_name, color_scales["YlGnBu"])
        
        colormap = branca.colormap.LinearColormap(
            colors=colors, 
            index=np.linspace(0, 1, len(colors)),
            vmin=0,
            vmax=1,
            caption='Probability of Viviparity'
        )
        folium_map.add_child(colormap)
    else:
        print("Warning: No valid GeoJSON features found for contour overlay")
        
    # Create a separate layer for points showing exact probability values
    points_layer = folium.FeatureGroup(name="Sample Points (toggle on/off)")
    
    # Add a subset of points with probability values (to avoid cluttering the map)
    # Use systematic sampling to get a representative distribution
    sample_size = min(500, len(df_pred))  # Limit to 500 points max
    step = max(1, len(df_pred) // sample_size)
    
    df_sample = df_pred.iloc[::step].dropna(subset=["Probability"])
    
    for _, point in df_sample.iterrows():
        # Skip points without valid probability
        if np.isnan(point.get("Probability", np.nan)):
            continue
            
        # Create color based on probability
        prob_color = plt.cm.YlGnBu(point["Probability"])
        # Convert RGBA to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(prob_color[0]*255), 
            int(prob_color[1]*255), 
            int(prob_color[2]*255)
        )
        
        # Create popup with detailed information
        popup_html = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Grid Point</b><br>
            Lat: {point.Latitude:.4f}<br>
            Lon: {point.Longitude:.4f}<br>
            <b>Probability:</b> {point.Probability:.3f}<br>
            <hr style="margin: 5px 0;">
            <b>Environmental Data:</b><br>
        """
        
        # Add environmental variables
        for var in [col for col in point.index if col not in ['Latitude', 'Longitude', 'Probability', 'distance']]:
            if not pd.isna(point[var]):
                popup_html += f"{var}: {point[var]:.1f}<br>"
        
        popup_html += "</div>"
        
        # Add circle marker
        folium.CircleMarker(
            location=[point.Latitude, point.Longitude],
            radius=2,  # Small points
            color=hex_color,
            fill=True,
            fill_color=hex_color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(points_layer)
    
    # Add the points layer to the map but set it to off by default
    points_layer.add_to(folium_map)
    
    # Create separate layers for viviparous and oviparous species
    viviparous_layer = folium.FeatureGroup(name="Viviparous Species (1)", show=True)
    oviparous_layer = folium.FeatureGroup(name="Oviparous Species (0)", show=True)
    unknown_layer = folium.FeatureGroup(name="Unknown Reproductive Mode", show=True)
    
    # Add real data points, colored by their reproductive mode
    for _, row in df_real.iterrows():
        if np.isnan(row.Latitude) or np.isnan(row.Longitude):
            continue
            
        # Build popup content
        popup_text = f"""
        <div style="font-family: Arial; font-size: 12px;">
            <b>Observation Point</b><br>
            Lat: {row.Latitude:.4f}<br>
            Lon: {row.Longitude:.4f}<br>
        """
        
        if 'Species' in row:
            popup_text += f"<b>Species:</b> {row.Species}<br>"
        
        if 'IsViviparous' in row:
            if row.IsViviparous == 1:
                popup_text += "<b>Reproductive Mode:</b> Viviparous<br>"
            elif row.IsViviparous == 0:
                popup_text += "<b>Reproductive Mode:</b> Oviparous<br>"
            else:
                popup_text += "<b>Reproductive Mode:</b> Unknown<br>"
                
        popup_text += "</div>"
                
        # Create marker with appropriate color
        if 'IsViviparous' in row and row.IsViviparous == 1:
            # Viviparous - use darker blue
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.9
            ).add_to(viviparous_layer)
        elif 'IsViviparous' in row and row.IsViviparous == 0:
            # Oviparous - use lighter yellow/orange
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='orange',
                fill=True,
                fill_color='orange',
                fill_opacity=0.9
            ).add_to(oviparous_layer)
        else:
            # Unknown - use red or gray
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=6,
                popup=folium.Popup(popup_text, max_width=300),
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.9
            ).add_to(unknown_layer)
    
    # Add all layers to the map
    viviparous_layer.add_to(folium_map)
    oviparous_layer.add_to(folium_map)
    unknown_layer.add_to(folium_map)
    
    # Add a layer for predicted vs actual match assessment
    if 'IsViviparous' in df_real.columns and df_real['IsViviparous'].isin([0, 1]).any():
        assessment_layer = folium.FeatureGroup(name="Prediction Assessment", show=False)
        
        # For each observation point with known reproductive mode
        for _, row in df_real[df_real['IsViviparous'].isin([0, 1])].iterrows():
            # Find nearest prediction point
            df_pred['temp_dist'] = np.sqrt(
                (df_pred['Latitude'] - row.Latitude)**2 + 
                (df_pred['Longitude'] - row.Longitude)**2
            )
            nearest_pred = df_pred.loc[df_pred['temp_dist'].idxmin()]
            
            # Skip if no valid prediction
            if pd.isna(nearest_pred.get('Probability', np.nan)):
                continue
                
            # Calculate match percentage
            if row.IsViviparous == 1:
                match_pct = nearest_pred.Probability * 100
                correct_pred = nearest_pred.Probability >= 0.5
            else:  # IsViviparous == 0
                match_pct = (1 - nearest_pred.Probability) * 100
                correct_pred = nearest_pred.Probability < 0.5
            
            # Determine color based on match quality
            if correct_pred:
                # Good prediction - use green with intensity based on confidence
                color = f'#{int(155 + match_pct):02x}ff{int(155):02x}'
            else:
                # Poor prediction - use red with intensity based on error
                color = f'#ff{int(155 + (100-match_pct)):02x}{int(155):02x}'
            
            # Add marker showing prediction quality
            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px;">
                <b>Prediction Assessment</b><br>
                Actual: {"Viviparous" if row.IsViviparous == 1 else "Oviparous"}<br>
                Predicted Probability: {nearest_pred.Probability:.3f}<br>
                <b>Match: {match_pct:.1f}%</b><br>
                <b>Outcome: {"✓ Correct" if correct_pred else "✗ Incorrect"}</b>
            </div>
            """
            
            folium.CircleMarker(
                location=[row.Latitude, row.Longitude],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.9
            ).add_to(assessment_layer)
        
        # Clean up temporary column
        if 'temp_dist' in df_pred.columns:
            df_pred.drop('temp_dist', axis=1, inplace=True)
            
        # Add the assessment layer to the map
        assessment_layer.add_to(folium_map)
        
        # Add a legend explaining the colors
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: 90px; 
                    border:2px solid grey; z-index:9999; font-size:12px;
                    background-color: white; padding: 10px;
                    border-radius: 5px;">
            <span style="color: blue;"><b>●</b></span> Viviparous Species (1)<br>
            <span style="color: orange;"><b>●</b></span> Oviparous Species (0)<br>
            <span style="color: red;"><b>●</b></span> Unknown Reproductive Mode<br>
            <hr style="margin: 5px 0;">
            <i>Toggle layers using the control panel</i>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(legend_html))

    # Add map controls
    folium.LayerControl(collapsed=False).add_to(folium_map)
    plugins.Fullscreen().add_to(folium_map)
    plugins.MeasureControl().add_to(folium_map)
    
    # Save the map
    try:
        folium_map.save("viviparity_map_regres.html")
        print("✅ Map saved as 'viviparity_map.html'")
        
        # Also save a more detailed version
        folium_map.save("viviparity_map_detailed_regress.html")
        print("✅ Detailed map saved as 'viviparity_map_detailed.html'")
    except Exception as e:
        print(f"Error saving map: {e}")

###########################
# ANALYZE ENVIRONMENTAL DATA
###########################
def analyze_environmental_data(df_grid, df_real=None):
    """Perform additional analysis on environmental data."""
    # Get all environmental variables
    env_vars = [col for col in df_grid.columns if col not in ['Latitude', 'Longitude', 'Probability']]
    
    if len(env_vars) == 0:
        print("⚠️ No environmental variables found for analysis")
        return
    
    print("\n📊 Analyzing environmental data patterns...")
    
    # Create a grid for visualization
    try:
        # For each environmental variable, calculate statistics by latitude band
        lat_bands = pd.cut(df_grid['Latitude'], bins=10)
        lat_analysis = df_grid.groupby(lat_bands, observed=False)[env_vars].mean()
        lat_analysis.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in lat_analysis.index]
        
        # Save latitude band analysis
        lat_analysis.to_csv("environmental_by_latitude.csv")
        print("✅ Latitude band analysis saved to environmental_by_latitude.csv")
        
        # For each environmental variable, calculate statistics by longitude band
        lon_bands = pd.cut(df_grid['Longitude'], bins=10)
        lon_analysis = df_grid.groupby(lon_bands, observed=False)[env_vars].mean()
        lon_analysis.index = [f"{int(interval.left)}-{int(interval.right)}" for interval in lon_analysis.index]
        
        # Save longitude band analysis
        lon_analysis.to_csv("environmental_by_longitude.csv")
        print("✅ Longitude band analysis saved to environmental_by_longitude.csv")
        
        # If we have real data points, compare environmental conditions at those points
        if df_real is not None and len(df_real) > 0:
            # For each real data point, extract nearest grid point's environmental data
            real_env_data = []
            
            for _, real_row in df_real.iterrows():
                # Calculate distance to each grid point
                df_grid['distance'] = np.sqrt(
                    (df_grid['Latitude'] - real_row['Latitude'])**2 + 
                    (df_grid['Longitude'] - real_row['Longitude'])**2
                )
                
                # Get closest point
                closest_idx = df_grid['distance'].idxmin()
                closest_point = df_grid.loc[closest_idx].copy()
                
                # Add real point info
                if 'Species' in real_row:
                    closest_point['Species'] = real_row['Species']
                
                # Add to collection
                real_env_data.append(closest_point)
            
            # Create DataFrame with environmental data at real points
            df_real_env = pd.DataFrame(real_env_data)
            
            # Save to CSV
            df_real_env.to_csv("environmental_at_real_points.csv", index=False)
            print("✅ Environmental data at real points saved to environmental_at_real_points.csv")
    
    except Exception as e:
        print(f"⚠️ Error in environmental analysis: {e}")

###########################
# MAIN
###########################
def main():
    try:
        print("Starting viviparity probability mapping...")
        
        # Load real data points
        df_real, min_lon, min_lat, max_lon, max_lat = load_real_data(CSV_FILE)
        
        # Load model
        model, scaler, features = load_model()
        
        # Create grid of points
        df_grid = make_lon_lat_grid(min_lon, min_lat, max_lon, max_lat, RESOLUTION)
        
        # Predict viviparity probability
        df_pred = predict_for_grid(df_grid, model, scaler, features)
        
        # Perform additional environmental analysis
        analyze_environmental_data(df_pred, df_real)
        
        # Create contours
        geojson_data = create_contours(df_pred)
        
        # Create interactive map
        create_folium_map(df_pred, df_real, geojson_data)
        
        print("\n✅ Process completed successfully!")
        print("📄 Output files:")
        print("  - environmental_values.csv (All environmental data by lat/long)")
        print("  - environmental_values_summary.csv (Summary statistics)")
        print("  - prediction_with_environmental_data.csv (Complete dataset with predictions)")
        print("  - environmental_by_latitude.csv (Environmental trends by latitude)")
        print("  - environmental_by_longitude.csv (Environmental trends by longitude)")
        print("  - environmental_at_real_points.csv (Environmental data at observation points)")
        print("  - viviparity_map.html (Interactive visualization)")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
