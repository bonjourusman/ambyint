import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, average_precision_score, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.utils.class_weight import compute_class_weight
import pickle
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('dataset/train.csv', parse_dates=['Timestamp'])

# Change data type of Monitoring_Alert to category:
df['Monitoring_Alert'] = df['Monitoring_Alert'].astype('category')

# Change data type of Maintenance_Code to text:
df['Maintenance_Code'] = df['Maintenance_Code'].fillna('None')
df['Maintenance_Code'] = df['Maintenance_Code'].astype(str)

###################################################
# Exploratory Data Analysis
###################################################

# Number of Wells
#print(f"Number of Wells: {df['Well_ID'].nunique()}\n")

# Data belongs to a single well only. 'Well_ID' column can thus be ignored:
df = df.drop(columns=['Well_ID'])

# Summary statistics and previews
#print(df.info(),'\n')
#print(df.describe(),'\n')
#print(df.head(),'\n')
#print(df.tail(),'\n')

selected_cols = ['Timestamp']

# CATEGORICAL FEATURE ANALYSIS:
# -----------------------------

# Feature 1: Well_Operating_Status

# 'Well_Operating_Status' appears to be interesting and could have some correlation with the response: Monitoring_Alert. Explore further:
#print(f"Well Operating Status counts:\n{df.Well_Operating_Status.value_counts()}\n")

# Evaluate 'Monitoring_Alert' by 'Well_Operating_Status':
#print(df.groupby('Well_Operating_Status')['Monitoring_Alert'].value_counts(), '\n')

# 'Monitoring_Alert' only gets triggered when the Operating Status is 'Operating'. It would make sense to ignore the other operating statuses (i.e. Maintenance, Standby, Startup) for the purpose of predictive modeling and to prevent bias:
df1 = df[df['Well_Operating_Status'] == 'Operating'].copy()

df1.sort_values(by='Timestamp', inplace=True)

# Feature 2: Operational_Notes

# 'Operational_Notes' has a relatively large number of null values. Explore further:
#print(f"Unique Operational Notes:\n{df1.Operational_Notes.value_counts()}\n")

# The values in the 'Operational_Notes' feature provide insights into well operations and should be analyzed further for patterns or anomalies. Check its relationship with 'Monitoring_Alert'
#print(df1.groupby('Operational_Notes')['Monitoring_Alert'].value_counts(), '\n')

# It appears that certain notes correlate with specific 'Monitoring_Alert' responses.
# Therefore, they may not be useful indicators for predicting an anomaly event.

# Feature 3: Lubrication_Status

# 'Lubrication_Status' has a relatively large number of null values. Explore further:
#print(f"Unique Lubrication Status:\n{df1.Lubrication_Status.value_counts()}\n")

# Check its relationship with 'Monitoring_Alert'
#print(df1.groupby('Lubrication_Status')['Monitoring_Alert'].value_counts(), '\n')

# 'Lubrication_Status' seems relevant to predicting response, though the best way to fill missing values or handle its null entries is unclear. Need more contextual information.

# Feature 4: Maintenance_Code

#print(f"Unique Maintenance Codes:\n{df1.Maintenance_Code.value_counts()}\n")

# Check Maintenance_Code relationship with 'Monitoring_Alert'
#print(df1.groupby('Maintenance_Code')['Monitoring_Alert'].value_counts(), '\n')

# 'Maintenance_Code' may be useful for predicting 'Monitoring_Alert', though more domain specific analysis is needed to confirm its significance and how it is derived:
selected_cols.append('Maintenance_Code')

# Response: Monitoring_Alert

# Check count of Monitoring_Alert categories as we expect it to have imbalanced class distribution
#print(f"Monitoring_Alert category counts:\n{df1['Monitoring_Alert'].value_counts()}\n")

# QUANTITATIVE (NUMERICAL) FEATURE ANALYSIS:
# -------------------------------------------

# Check range of values for Liquid_Flow_Rate_m3d:
#print(f"Range of Liquid_Flow_Rate_m3d:\n{df1['Liquid_Flow_Rate_m3d'].min()} to {df1['Liquid_Flow_Rate_m3d'].max()}\n")

# The flow rate contains negative values, which should be investigated as they could mean potential loss of production or data entry/sensor errors:
#print(f"Number of negative values in Liquid_Flow_Rate_m3d:\n{(df1['Liquid_Flow_Rate_m3d'] < 0).sum()}\n")

# Explore the data when the Liquid_Flow_Rate_m3d is negative:
negative_flow_data = df1[df1['Liquid_Flow_Rate_m3d'] < 0]
#print(f"Details of rows with negative Liquid_Flow_Rate_m3d:\n{negative_flow_data}\n")

# Check monitoring alert when Liquid_Flow_Rate_m3d is negative:
monitoring_alert_negative_flow = negative_flow_data['Monitoring_Alert'].value_counts()
#print(f"Monitoring_Alert distribution for negative Liquid_Flow_Rate_m3d:\n{monitoring_alert_negative_flow}\n")

# Explore the data when Liquid_Flow_Rate_m3d is an outlier:
outlier_flow_data = df1[df1['Liquid_Flow_Rate_m3d'] > df1['Liquid_Flow_Rate_m3d'].quantile(0.99)]
#print(f"Details of rows with outlier Liquid_Flow_Rate_m3d:\n{outlier_flow_data}\n")

# Check monitoring alert when Liquid_Flow_Rate_m3d is an outlier
monitoring_alert_outlier_flow = outlier_flow_data['Monitoring_Alert'].value_counts()
#print(f"Monitoring_Alert distribution for outlier Liquid_Flow_Rate_m3d:\n{monitoring_alert_outlier_flow}\n")

# Additional analysis or visualizations to further explore the dataset:
'''
# Visualize the marginal distribution of Liquid_Flow_Rate_m3d to identify patterns or anomalies
sns.histplot(df1['Liquid_Flow_Rate_m3d'], bins=50, kde=True)
plt.title('Distribution of Liquid_Flow_Rate_m3d')
plt.xlabel('Liquid_Flow_Rate_m3d')
plt.ylabel('Frequency')
plt.savefig('plots/liquid_flow_rate_distribution.png', dpi=300)
plt.close()

# Visualize the conditional distribution of Liquid_Flow_Rate_m3d based on Monitoring_Alert
sns.boxplot(x='Monitoring_Alert', y='Liquid_Flow_Rate_m3d', data=df1)
plt.title('Conditional Distribution of Liquid_Flow_Rate_m3d by Monitoring_Alert')
plt.xlabel('Monitoring_Alert')
plt.ylabel('Liquid_Flow_Rate_m3d')
plt.savefig('plots/liquid_flow_rate_cond_distribution.png', dpi=300)
plt.close()
'''
# It is evident from the plot above, that the negative flow rates are outliers.
# Considering they represent a very small percent of the data, and assuming they are factually correct, we'll keep the rows (with negative flow rate) in the model as-is.
# When modeling, however, it might be useful to evaluate them as potential outliers or leverage points affecting the quality of model fit.

# Similar marginal and conditional analysis can be performed on other quantitative features.

# Next, generate a heatmap to visualize the correlation among quantitative features
quant_features = df1.select_dtypes(include=['float64', 'int64']).columns

'''
plt.figure(figsize=(12, 10))
sns.heatmap(df1[quant_features].corr(), 
           annot=True, 
           cmap='coolwarm_r', 
           fmt='.2f',
           annot_kws={'size': 9})
plt.title('Correlation Heatmap of Quantitative Features', fontsize=14)
plt.savefig('plots/heatmap_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
'''
# 'Casing_Pressure_psi' and 'Casing_Pressure_SensorB_psi' have a strong positive correlation of 0.8, indicating potential redundancy. We'll ignore one of these for modeling:
quant_features_selected = [f for f in quant_features if f != 'Casing_Pressure_SensorB_psi']

# Fill null values of selected quantitative features using average of previous 3 and next 3 observations:
df1[quant_features_selected] = df1[quant_features_selected].fillna(df1[quant_features_selected].rolling(window=7, min_periods=1, center=True).mean())

selected_cols += quant_features_selected
#print(f"Selected columns: {selected_cols}")

###################################################
# Data Prep for Model Training
###################################################

df_modeling = df1[selected_cols].copy()
df_modeling['Monitoring_Alert'] = df1['Monitoring_Alert'].copy() # add response column

# Define periods to capture temporal effect for each observation
NUM_LAG_PERIODS = 3
NUM_DELTA_PERIODS = 3

def add_lag_and_delta_features(df, num_lags, num_deltas):

    df_copy = df.copy()

    # Add lag & delta features based on set periods for each quantitative feature:
    lag_delta_features = df_copy.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Create all lag and delta features at once to avoid DataFrame fragmentation
    lag_delta_dfs = []

    for feature in lag_delta_features:
        feature_dfs = []

        # Create lag features
        for lag in range(1, num_lags + 1):
            lag_series = df_copy[feature].shift(lag)
            lag_series.name = f'{feature}_lag_{lag}'
            feature_dfs.append(lag_series)

        # Create delta features
        for delta in range(1, num_deltas + 1):
            delta_series = df_copy[feature].diff(delta)
            delta_series.name = f'{feature}_delta_{delta}'
            feature_dfs.append(delta_series)

        # Combine all features for this column
        if feature_dfs:
            lag_delta_dfs.extend(feature_dfs)

    # Concatenate all new features
    if lag_delta_dfs:
        new_features_df = pd.concat(lag_delta_dfs, axis=1)
        df_copy = pd.concat([df_copy, new_features_df], axis=1)
    
    return df_copy

df_modeling = add_lag_and_delta_features(df_modeling, NUM_LAG_PERIODS, NUM_DELTA_PERIODS)

# Drop rows with null values
df_modeling.dropna(inplace=True)
df_modeling.reset_index(drop=True, inplace=True)

# Convert categorical features to indicator/dummy variables
# Drop a specific category by specifying dummy_na and then dropping manually
df_modeling = pd.get_dummies(df_modeling, columns=['Maintenance_Code'], drop_first=False)
df_modeling = df_modeling.drop('Maintenance_Code_None', axis=1)  # Dropping the 'None' category

# Drop the Timestamp column as its not required for modeling
df_modeling = df_modeling.drop('Timestamp', axis=1)

# QA:
#print(df_modeling[df_modeling.columns[df_modeling.columns.str.startswith('VFD')]].iloc[:30])
#print(df_modeling['VFD_Speed_pct_delta_5'].tolist()[:1000])

# Summary statistics
#print(df_modeling.info(),'\n')
#print(df_modeling.describe(),'\n')
#print(df_modeling.head(),'\n')
#print(df_modeling.tail(),'\n')

# Separate features and response variable
X = df_modeling.drop('Monitoring_Alert', axis=1)  
y = df_modeling['Monitoring_Alert']

# Split data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Input data shape
print(f"Training set shape: {X_train.shape},\nValidation set shape: {X_valid.shape}\n")

###################################################
# Model Training
###################################################

# Test multiple models and evaluate their performance. The models include Logistic Regression, SVMs, Neural Networks, Decision Trees, XGBoost, and Random Forests.

print("Class distribution in training set:\n")
class_counts = Counter(y_train)
print(f"Class 0: {class_counts[0]} samples")
print(f"Class 1: {class_counts[1]} samples")
print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")

# Calculate class weights for weighted models
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Calculated class weights: {class_weight_dict}")

# Define sampling strategies
sampling_strategies = {
    'Original': None,
    'SMOTE': SMOTE(random_state=42),
    #'ADASYN': ADASYN(random_state=42),
    'RandomOverSampler': RandomOverSampler(random_state=42),
    #'SMOTETomek': SMOTETomek(random_state=42),
    #'SMOTEENN': SMOTEENN(random_state=42)
}

# Define hyperparameter grids for each model (with class imbalance handling)
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced', class_weight_dict]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'class_weight': [None, 'balanced', class_weight_dict]
    },
    'Neural Network': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', class_weight_dict]
    },
    'XGBoost': {
        'n_estimators': [100], # [50, 100, 200]
        'max_depth': [5], # [3, 5, 7]
        'learning_rate': [0.05], # [0.01, 0.05, 0.1, 0.2]
        'subsample': [0.9], # [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, class_counts[0]/class_counts[1]]
    },
    'Random Forest': {
        'n_estimators': [100], # [50, 100, 200]
        'max_depth': [10], # [5, 10, 15, None]
        'min_samples_split': [5], # [2, 5, 10]
        'min_samples_leaf': [2], # [1, 2, 4]
        'class_weight': [None, 'balanced', class_weight_dict]
    }
}

# Initialize base models with class imbalance handling
base_models = {
    #'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    #'SVM': SVC(random_state=42, probability=True),
    #'Neural Network': MLPClassifier(random_state=42, max_iter=1000),
    #'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
    'Random Forest': RandomForestClassifier(random_state=42)
}

# Store results
results = {}
best_models = {}

# Define multiple scoring metrics for imbalanced datasets
scoring_metrics = ['f1', 'precision', 'recall', 'roc_auc']

# Train and evaluate each model with different sampling strategies
for sampling_name, sampler in sampling_strategies.items():
    print(f"\n{'='*80}")
    print(f"TRAINING WITH SAMPLING STRATEGY: {sampling_name}")
    print(f"{'='*80}")

    # Apply sampling strategy
    if sampler is not None:
        try:
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train.values, y_train.values)
            print(f"After {sampling_name}:")
            resampled_counts = Counter(y_train_resampled)
            print(f"Class 0: {resampled_counts[0]} samples")
            print(f"Class 1: {resampled_counts[1]} samples")
        except Exception as e:
            print(f"Error with {sampling_name}: {e}")
            continue
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    # Train each model with current sampling strategy
    for name, base_model in base_models.items():
        model_key = f"{name}_{sampling_name}"
        print(f"\nTuning hyperparameters for {name} with {sampling_name}...")

        # Stratified k-fold for imbalanced datasets
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Perform grid search with cross-validation using multiple metrics
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grids[name],
            cv=skf,
            scoring='f1_micro',  # Primary metric for model selection
            n_jobs=-1,
            verbose=0
        )

        # Fit grid search
        grid_search.fit(X_train_resampled, y_train_resampled)

        # Get best model
        best_model = grid_search.best_estimator_
        best_models[model_key] = best_model

        # Make predictions with best model
        y_pred = best_model.predict(X_valid)

        # Get prediction probabilities for AUC calculation
        if hasattr(best_model, "predict_proba"):
            y_pred_proba = best_model.predict_proba(X_valid)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_pred_proba = best_model.decision_function(X_valid)
        else:
            y_pred_proba = y_pred

        # Calculate all metrics
        accuracy = accuracy_score(y_valid, y_pred)
        precision = precision_score(y_valid, y_pred, zero_division=0)
        recall = recall_score(y_valid, y_pred, zero_division=0)
        f1 = f1_score(y_valid, y_pred, average='micro', zero_division=0)

        # AUC calculation
        try:
            auc = roc_auc_score(y_valid, y_pred_proba)
        except ValueError:
            auc = 0.0

        # Additional metrics for imbalanced data
        balanced_acc = balanced_accuracy_score(y_valid, y_pred)
        avg_precision = average_precision_score(y_valid, y_pred_proba)

        # Cross-validation scores with multiple metrics
        cv_f1_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=skf, scoring='f1_micro')
        cv_precision_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=skf, scoring='precision')
        cv_recall_scores = cross_val_score(best_model, X_train_resampled, y_train_resampled, cv=skf, scoring='recall')

        # Store results
        results[model_key] = {
            'model_name': name,
            'sampling_strategy': sampling_name,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'avg_precision': avg_precision,
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'cv_precision_mean': cv_precision_scores.mean(),
            'cv_recall_mean': cv_recall_scores.mean(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_
        }

        print(f"{name} - Best Parameters: {grid_search.best_params_}")
        print(f"{name} - Best CV F1-Score: {grid_search.best_score_:.4f}")
        print(f"{name} - Valid Accuracy: {accuracy:.4f}")
        print(f"{name} - Valid Balanced Accuracy: {balanced_acc:.4f}")
        print(f"{name} - Valid Precision: {precision:.4f}")
        print(f"{name} - Valid Recall: {recall:.4f}")
        print(f"{name} - Valid F1-Score: {f1:.4f}")
        print(f"{name} - Valid AUC: {auc:.4f}")
        print(f"{name} - Valid Avg Precision: {avg_precision:.4f}")
        print(f"{name} - CV F1-Score: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")

# Create comprehensive results summary
results_df = pd.DataFrame({
    'Model': [results[model]['model_name'] for model in results.keys()],
    'Sampling': [results[model]['sampling_strategy'] for model in results.keys()],
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Balanced_Acc': [results[model]['balanced_accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1_score'] for model in results.keys()],
    'AUC': [results[model]['auc'] for model in results.keys()],
    'Avg_Precision': [results[model]['avg_precision'] for model in results.keys()],
    'CV_F1_Mean': [results[model]['cv_f1_mean'] for model in results.keys()],
    'CV_F1_Std': [results[model]['cv_f1_std'] for model in results.keys()],
    'Model_Key': list(results.keys())
})

# Sort by F1-Score (primary metric for imbalanced datasets)
results_df = results_df.sort_values('F1-Score', ascending=False)
print("\n" + "="*120)
print("COMPREHENSIVE MODEL PERFORMANCE SUMMARY (SORTED BY F1-SCORE)")
print("="*120)
print(results_df.drop('Model_Key', axis=1).to_string(index=False))

# Find best performing model based on F1-Score
best_model_key = results_df.iloc[0]['Model_Key']
best_model = best_models[best_model_key]
best_result = results[best_model_key]

print(f"\n{'='*80}")
print("BEST PERFORMING MODEL DETAILS")
print(f"{'='*80}")
print(f"Model: {best_result['model_name']}")
print(f"Sampling Strategy: {best_result['sampling_strategy']}")
print(f"F1-Score: {best_result['f1_score']:.4f}")
print(f"Accuracy: {best_result['accuracy']:.4f}")
print(f"Balanced Accuracy: {best_result['balanced_accuracy']:.4f}")
print(f"Precision: {best_result['precision']:.4f}")
print(f"Recall: {best_result['recall']:.4f}")
print(f"AUC: {best_result['auc']:.4f}")
print(f"Average Precision: {best_result['avg_precision']:.4f}")
print(f"CV F1-Score: {best_result['cv_f1_mean']:.4f} (+/- {best_result['cv_f1_std'] * 2:.4f})")
print(f"Best parameters: {best_result['best_params']}")

# Show top 5 models for comparison
print(f"\n{'='*80}")
print("TOP 5 MODELS COMPARISON")
print(f"{'='*80}")
top_5_df = results_df.head(5)[['Model', 'Sampling', 'F1-Score', 'Precision', 'Recall', 'AUC', 'Balanced_Acc']]
print(top_5_df.to_string(index=False))

# Performance by sampling strategy
print(f"\n{'='*80}")
print("PERFORMANCE BY SAMPLING STRATEGY")
print(f"{'='*80}")
sampling_summary = results_df.groupby('Sampling').agg({
    'F1-Score': ['mean', 'max'],
    'Precision': ['mean', 'max'],
    'Recall': ['mean', 'max'],
    'AUC': ['mean', 'max']
}).round(4)
sampling_summary.columns = ['_'.join(col).strip() for col in sampling_summary.columns]
print(sampling_summary)

print(f"\n{'='*80}")
print("CONFUSION MATRICES FOR TOP 3 MODELS")
print(f"{'='*80}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (_, row) in enumerate(results_df.head(3).iterrows()):
    model_key = row['Model_Key']
    y_pred = results[model_key]['predictions']

    cm = confusion_matrix(y_valid, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
    axes[i].set_title(f"{row['Model']} - {row['Sampling']}\nF1-Score (micro): {row['F1-Score']:.3f}")
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('plots/heatmap_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

###################################################
# Save Best Model
###################################################

with open('model.pkl','wb') as f:
    pickle.dump(best_model,f)

###################################################
# Model Evaluation on Test Set
###################################################

# Load Test Data
df_test = pd.read_csv('dataset/test.csv', parse_dates=['Timestamp'])

# Perform transformations to match format with Training Data
# ----------------------------------------------------------------

# Change data type of Maintenance_Code to text:
df_test['Maintenance_Code'] = df_test['Maintenance_Code'].fillna('None')
df_test['Maintenance_Code'] = df_test['Maintenance_Code'].astype(str)

df_test1 = df_test[df_test['Well_Operating_Status'] == 'Operating'].copy()

df_test1.sort_values(by='Timestamp', inplace=True)

df_test1[quant_features_selected] = df_test1[quant_features_selected].fillna(df_test1[quant_features_selected].rolling(window=7, min_periods=1, center=True).mean())

df_test2 = df_test1[selected_cols].copy()

df_test2 = add_lag_and_delta_features(df_test2, NUM_LAG_PERIODS, NUM_DELTA_PERIODS)

df_test2.dropna(inplace=True)
df_test2.reset_index(drop=True, inplace=True)

df_test2 = pd.get_dummies(df_test2, columns=['Maintenance_Code'], drop_first=False)
df_test2 = df_test2.drop('Maintenance_Code_None', axis=1)  # Dropping the 'None' category

X_test = df_test2[[c for c in df_test2.columns if c != 'Timestamp']]

# Load Best Model obtained from training
# ----------------------------------------------------------------
with open('model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Generate predictions
y_test = best_model.predict(X_test)

# Merge predictions with the transformed test data
df_test_results = pd.concat([df_test2, pd.Series(y_test, name='Predicted_Monitoring_Alert')], axis=1)

# Join the test results with raw test data on Timestamp column
df_final_results = pd.merge(df_test, df_test_results, on='Timestamp', how='left')

# For rows with Operating Status != Operational & those excluded because of min lag/delta periods, manually set predictions to 0
df_final_results['Predicted_Monitoring_Alert'] = df_final_results['Predicted_Monitoring_Alert'].fillna(0)

# Final formatting
df_final_results['Predicted_Monitoring_Alert'] = df_final_results['Predicted_Monitoring_Alert'].astype(int)

# Export predictions to CSV
df_final_results['Predicted_Monitoring_Alert'].to_csv('test_predictions.csv', index=False)

print('End of script execution.')