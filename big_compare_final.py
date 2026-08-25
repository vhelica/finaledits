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

