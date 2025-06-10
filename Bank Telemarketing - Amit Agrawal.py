# =============================================================================
# Import all necessary libraries
# =============================================================================

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import logging
import os
import sys
from pathlib import Path

# =============================================================================
# Setup logging
# =============================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bank_telemarketing_analysis.log')
    ]
)
logger = logging.getLogger('bank_telemarketing')

# =============================================================================
# Import the raw data
# =============================================================================

try:
    file_path = r'C:\Users\Amit Agrawal\Dropbox\QL Case Study\bank-additional-full.xlsx'
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}")
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    logger.info(f"Loading data from {file_path}")
    raw_data = pd.read_excel(file_path)
    logger.info(f"Successfully loaded data with {raw_data.shape[0]} rows and {raw_data.shape[1]} columns")
    raw_data.info()

    # Rename fields
    try:
        raw_data.rename(columns = {'cons.price.idx':'cons_price_idx', 'cons.conf.idx':'cons_conf_idx', 'emp.var.rate':'emp_var_rate', 'nr.employed':'nr_employed'}, inplace = True)
        logger.info("Successfully renamed columns")
    except Exception as e:
        logger.error(f"Error renaming columns: {str(e)}")
        raise
        
except FileNotFoundError as e:
    logger.error(f"File not found error: {str(e)}")
    print(f"Error: {str(e)}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    print(f"Error loading data: {str(e)}")
    sys.exit(1)

# =============================================================================
# Scoping
# =============================================================================

# We will be building the model as a decision making tool before campaign starts, hence not use the featues related to current campaign

try:
    logger.info("Dropping features related to current campaign")
    raw_data = raw_data.drop(['contact', 'month', 'day_of_week', 'duration', 'campaign'], axis = 1)
    logger.info(f"Remaining columns after dropping: {list(raw_data.columns)}")
    raw_data.dtypes
except KeyError as e:
    logger.error(f"Error dropping columns, column not found: {str(e)}")
    print(f"Error: Column not found when dropping features: {str(e)}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error in data scoping: {str(e)}")
    print(f"Unexpected error in data scoping: {str(e)}")
    sys.exit(1)

# =============================================================================
# Exploratory Data Analysis
# =============================================================================

try:
    logger.info("Starting exploratory data analysis")
    
    # Univariate distribution of dependent variable
    try:
        logger.info("Analyzing target variable distribution")
        if 'y' not in raw_data.columns:
            logger.error("Target variable 'y' not found in dataset")
            raise KeyError("Target variable 'y' not found in dataset")
            
        print(raw_data['y'].value_counts())
        plt.figure()
        raw_data['y'].value_counts().plot.bar()
        plt.savefig('target_variable_distribution.png')
        logger.info("Generated target variable distribution plot")
        plt.show()
    except KeyError as e:
        logger.error(f"Key error in target variable analysis: {str(e)}")
        print(f"Error: {str(e)}")
        # Continue execution with available variables
    except Exception as e:
        logger.error(f"Error analyzing target variable: {str(e)}")
        print(f"Warning: Could not analyze target variable: {str(e)}")
        # Continue execution

    # Distribution of all numeric variables
    try:
        logger.info("Analyzing numeric variables")
        numeric_vars = ['age', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'euribor3m']
        
        # Check which variables are available
        available_numeric_vars = [var for var in numeric_vars if var in raw_data.columns]
        if not available_numeric_vars:
            logger.warning("No numeric variables found for analysis")
        else:
            logger.info(f"Analyzing distributions for: {available_numeric_vars}")
            raw_data[available_numeric_vars].describe()
    except Exception as e:
        logger.error(f"Error analyzing numeric variables: {str(e)}")
        print(f"Warning: Could not analyze numeric variables: {str(e)}")
        # Continue execution

    # Distribution of all categorical variables
    categorical_vars = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'pdays', 'previous', 'poutcome']
    
    for var in categorical_vars:
        try:
            if var not in raw_data.columns:
                logger.warning(f"Categorical variable '{var}' not found in dataset")
                continue
                
            logger.info(f"Analyzing distribution of '{var}'")
            print(raw_data[var].value_counts())
            plt.figure()
            raw_data[var].value_counts().plot.bar()
            plt.savefig(f'{var}_distribution.png')
            logger.info(f"Generated distribution plot for '{var}'")
            plt.show()
        except Exception as e:
            logger.error(f"Error analyzing categorical variable '{var}': {str(e)}")
            print(f"Warning: Could not analyze '{var}': {str(e)}")
            # Continue with next variable

    # Outlier treatment
    try:
        logger.info("Starting outlier treatment")
        
        # Age outlier treatment
        if 'age' in raw_data.columns:
            try:
                logger.info("Treating outliers in 'age' variable")
                plt.figure()
                print(raw_data['age'].plot.hist())
                age_p95 = raw_data['age'].quantile(0.95)
                print(f"95th percentile of age: {age_p95}")
                
                # Count outliers before treatment
                outliers_count = (raw_data['age'] > age_p95).sum()
                logger.info(f"Found {outliers_count} outliers in 'age' (>{age_p95})")
                
                # Apply treatment
                raw_data.age.loc[raw_data.age > age_p95] = age_p95
                logger.info(f"Capped 'age' values at {age_p95}")
                
                plt.figure()
                print(raw_data['age'].plot.hist())
                plt.savefig('age_after_outlier_treatment.png')
                logger.info("Generated histogram for 'age' after outlier treatment")
                plt.show()
            except Exception as e:
                logger.error(f"Error treating outliers in 'age': {str(e)}")
                print(f"Warning: Could not treat outliers in 'age': {str(e)}")
        
        # Display histograms for other numeric variables
        for var in ['cons_price_idx', 'cons_conf_idx']:
            if var in raw_data.columns:
                try:
                    plt.figure()
                    print(raw_data[var].plot.hist())
                    plt.savefig(f'{var}_histogram.png')
                    logger.info(f"Generated histogram for '{var}'")
                    plt.show()
                except Exception as e:
                    logger.error(f"Error generating histogram for '{var}': {str(e)}")
        
        # Euribor3m outlier treatment
        if 'euribor3m' in raw_data.columns:
            try:
                logger.info("Treating outliers in 'euribor3m' variable")
                plt.figure()
                print(raw_data['euribor3m'].plot.hist())
                euribor3m_p05 = raw_data['euribor3m'].quantile(0.05)
                print(f"5th percentile of euribor3m: {euribor3m_p05}")
                
                # Count outliers before treatment
                outliers_count = (raw_data['euribor3m'] < euribor3m_p05).sum()
                logger.info(f"Found {outliers_count} outliers in 'euribor3m' (<{euribor3m_p05})")
                
                # Apply treatment
                raw_data.euribor3m.loc[raw_data.euribor3m < euribor3m_p05] = euribor3m_p05
                logger.info(f"Floor applied to 'euribor3m' values at {euribor3m_p05}")
                
                plt.figure()
                print(raw_data['euribor3m'].plot.hist())
                plt.savefig('euribor3m_after_outlier_treatment.png')
                logger.info("Generated histogram for 'euribor3m' after outlier treatment")
                plt.show()
            except Exception as e:
                logger.error(f"Error treating outliers in 'euribor3m': {str(e)}")
                print(f"Warning: Could not treat outliers in 'euribor3m': {str(e)}")
    except Exception as e:
        logger.error(f"Error in outlier treatment: {str(e)}")
        print(f"Warning: Could not complete outlier treatment: {str(e)}")
        # Continue execution

    # Correlation check
    try:
        logger.info("Generating correlation matrix")
        corrMatrix = raw_data.corr()
        print(corrMatrix)
        
        plt.figure(figsize=(12, 10))
        sn.heatmap(corrMatrix, annot=True)
        plt.savefig('correlation_matrix.png')
        logger.info("Generated correlation matrix heatmap")
        plt.show()
    except Exception as e:
        logger.error(f"Error generating correlation matrix: {str(e)}")
        print(f"Warning: Could not generate correlation matrix: {str(e)}")
        # Continue execution
        
    logger.info("Exploratory data analysis completed")
    
except Exception as e:
    logger.error(f"Error in exploratory data analysis: {str(e)}")
    print(f"Error in exploratory data analysis: {str(e)}")
    # Continue execution to allow other parts of the script to run

# =============================================================================
# Data Preprocessing
# =============================================================================

try:
    logger.info("Starting data preprocessing")
    
    # Create classifier
    try:
        logger.info("Creating classifier from target variable")
        df_one = pd.get_dummies(raw_data["y"])
        
        # Binary Data is Concatenated into Dataframe 
        df_two = pd.concat((df_one, raw_data), axis=1) 
        
        # We want no =0 and yes =1 So we drop no colunm here 
        if "no" not in df_two.columns:
            logger.warning("Column 'no' not found when trying to drop it")
        else:
            df_two = df_two.drop(["no"], axis=1)
            
        # Rename the Column 
        df_classifier = df_two.rename(columns={"yes": "classifier"})
        
        if "yes" not in df_two.columns:
            logger.warning("Column 'yes' not found when trying to rename to 'classifier'")
            
        df_classifier['classifier'] = df_classifier['classifier'].astype(str)
        logger.info("Successfully created classifier variable")
    except KeyError as e:
        logger.error(f"Key error in classifier creation: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating classifier: {str(e)}")
        raise

    # Create previous flag
    try:
        logger.info("Creating previous flag")
        df_classifier['previous_flag'] = np.where(df_classifier['previous'] > 0, 1, 0)
        df_classifier['previous_flag'] = df_classifier['previous_flag'].astype(str)
        logger.info("Successfully created previous flag")
    except KeyError as e:
        logger.error(f"Column 'previous' not found: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating previous flag: {str(e)}")
        raise

    # Feature engineering
    try:
        logger.info("Starting feature engineering for job segmentation")
        job_mappings = {
            'housemaid': 'employed', 'services': 'employed', 'admin.': 'employed',
            'blue-collar': 'employed', 'technician': 'employed', 'management': 'employed',
            'self-employed': 'employed', 'entrepreneur': 'employed', 'retired': 'retired',
            'unemployed': 'unemployed', 'unknown': 'unknown', 'student': 'student'
        }
        
        for job, segment in job_mappings.items():
            df_classifier.loc[df_classifier['job'] == job, 'job_seg'] = segment
            
        logger.info("Completed job segmentation")
        
        logger.info("Starting feature engineering for education segmentation")
        education_mappings = {
            'basic.4y': 'basic', 'basic.6y': 'basic', 'basic.9y': 'basic',
            'high.school': 'high_school', 'professional.course': 'higher_studies',
            'university.degree': 'higher_studies', 'unknown': 'unknown', 'illiterate': 'illiterate'
        }
        
        for edu, segment in education_mappings.items():
            df_classifier.loc[df_classifier['education'] == edu, 'edu_seg'] = segment
            
        logger.info("Completed education segmentation")
    except KeyError as e:
        logger.error(f"Column not found during feature engineering: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

    # Create the input data for logistics regression modelling
    try:
        logger.info("Creating input data for logistic regression modeling")
        required_columns = ['age', 'job_seg', 'marital', 'edu_seg', 'default', 'housing', 'loan', 
                           'poutcome', 'classifier', 'previous_flag', 'cons_price_idx', 'cons_conf_idx', 'euribor3m']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df_classifier.columns]
        if missing_columns:
            raise KeyError(f"Missing columns for model data: {missing_columns}")
            
        model_df = df_classifier[required_columns]
        logger.info("Successfully created input data frame for modeling")
    except KeyError as e:
        logger.error(f"Missing columns for model data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error creating input data for modeling: {str(e)}")
        raise

    # Create dummy variables
    try:
        logger.info("Creating dummy variables")
        dummy_columns = ['job_seg', 'marital', 'edu_seg', 'default', 'housing', 'loan', 'poutcome']
        model_df = pd.get_dummies(model_df, columns=dummy_columns)
        model_df = model_df.apply(pd.to_numeric)
        logger.info(f"Successfully created dummy variables. Final model shape: {model_df.shape}")
        model_df.dtypes
    except Exception as e:
        logger.error(f"Error creating dummy variables: {str(e)}")
        raise
        
except Exception as e:
    logger.error(f"Error in data preprocessing: {str(e)}")
    print(f"Error in data preprocessing: {str(e)}")
    sys.exit(1)

# =============================================================================
# Logistic Forecast Model
# =============================================================================

try:
    logger.info("Starting logistic regression modeling")
    
    # Run Logistics Regression to identify the significant predictors using smf
    try:
        logger.info("Fitting initial logistic regression model with all variables")
        # Drop all unknown columns from all categorical fields
        formula = ("classifier~ age + previous_flag + cons_price_idx + cons_conf_idx + euribor3m + "
                  "C(job_seg_employed) + C(job_seg_retired) + C(job_seg_student) + C(job_seg_unemployed) + "
                  "C(edu_seg_basic) + C(edu_seg_high_school) + C(edu_seg_higher_studies) + C(edu_seg_illiterate) + "
                  "C(marital_divorced) + C(marital_married) + C(marital_single) + "
                  "C(default_no) + C(default_yes) + C(housing_no) + C(housing_yes) + "
                  "C(loan_no) + C(loan_yes) + C(poutcome_failure) + C(poutcome_success)")
                  
        lr_model = smf.logit(formula=formula, data=model_df).fit()
        logger.info("Successfully fit initial logistic regression model")
        lr_model.summary()
    except Exception as e:
        logger.error(f"Error fitting initial logistic regression model: {str(e)}")
        raise

    # Create model using sklearn
    try:
        logger.info("Creating final logistic regression model with significant features")
        
        # Keep only significant factors based on the p values (<=0.05) of the above results
        # and split the data into x and y datasets
        significant_features = ['job_seg_retired', 'job_seg_student', 'edu_seg_basic', 
                              'default_no', 'cons_price_idx', 'cons_conf_idx', 'euribor3m']
        
        # Check if all significant features exist in model_df
        missing_features = [col for col in significant_features if col not in model_df.columns]
        if missing_features:
            raise KeyError(f"Missing significant features for model: {missing_features}")
            
        y_data = model_df['classifier']
        x_data = model_df[significant_features]
        
        logger.info("Successfully prepared data for final model")
    except KeyError as e:
        logger.error(f"Error selecting significant features: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error preparing data for final model: {str(e)}")
        raise

    # Split the data into training and test
    try:
        logger.info("Splitting data into training and test sets")
        x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42)
        logger.info(f"Training data shape: {x_training_data.shape}, Test data shape: {x_test_data.shape}")
    except Exception as e:
        logger.error(f"Error splitting data: {str(e)}")
        raise

    # Create and train the Logistic Regression Model
    try:
        logger.info("Training logistic regression model")
        # There is class imbalance problem, hence using class_weight='balanced'
        model = LogisticRegression(class_weight='balanced')
        model.fit(x_training_data, y_training_data)
        logger.info("Successfully trained logistic regression model")
        
        # Generate model coefficients
        model.coef_
        
        # Make predictions
        logger.info("Generating predictions on test data")
        predictions = model.predict(x_test_data)
        logger.info("Successfully generated predictions")
    except Exception as e:
        logger.error(f"Error training logistic regression model: {str(e)}")
        raise

    # Measure performance
    try:
        logger.info("Evaluating model performance")
        
        # Print classification report and confusion matrix
        print(classification_report(y_test_data, predictions))
        print(confusion_matrix(y_test_data, predictions))
        
        # Calculate various performance metrics
        acc = accuracy_score(y_test_data, predictions)
        bal_acc = balanced_accuracy_score(y_test_data, predictions)
        prec = precision_score(y_test_data, predictions)
        rec = recall_score(y_test_data, predictions)
        auc_score = roc_auc_score(y_test_data, predictions)
        f1 = f1_score(y_test_data, predictions)
        
        logger.info(f"Model performance metrics - Accuracy: {acc:.4f}, Balanced Accuracy: {bal_acc:.4f}, "
                    f"Precision: {prec:.4f}, Recall: {rec:.4f}, AUC: {auc_score:.4f}, F1: {f1:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating model performance: {str(e)}")
        raise

    # Plot ROC curve
    try:
        logger.info("Plotting ROC curve")
        fpr, tpr, _ = metrics.roc_curve(y_test_data, predictions)
        auc = metrics.roc_auc_score(y_test_data, predictions)
        plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.savefig('logistic_regression_roc_curve.png')  # Save the figure before showing
        logger.info(f"ROC curve plot saved to logistic_regression_roc_curve.png")
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")
        print(f"Warning: Could not create ROC curve plot: {str(e)}")
        # Continue execution even if plotting fails

    # Check for p values of the selected variables
    try:
        logger.info("Calculating p-values for selected variables")
        logit_model = sm.Logit(y_training_data, x_training_data)
        result = logit_model.fit()
        print(result.summary())
        logger.info("Successfully calculated p-values")
    except Exception as e:
        logger.error(f"Error calculating p-values: {str(e)}")
        print(f"Warning: Could not calculate p-values: {str(e)}")
        # Continue execution even if p-value calculation fails
        
except Exception as e:
    logger.error(f"Error in logistic regression modeling: {str(e)}")
    print(f"Error in logistic regression modeling: {str(e)}")
    # Don't exit here to allow for potential recovery or continuation with Random Forest

# =============================================================================
# Random Forest Model
# =============================================================================

try:
    logger.info("Starting Random Forest modeling")
    
    # Split the data into x and y datasets
    try:
        logger.info("Preparing data for Random Forest model")
        y_data = model_df['classifier']
        x_data = model_df.drop('classifier', axis=1)
        logger.info(f"Feature data shape: {x_data.shape}")
    except KeyError as e:
        logger.error(f"Column error when preparing Random Forest data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error preparing data for Random Forest: {str(e)}")
        raise

    # Split the data into training and test
    try:
        logger.info("Splitting data into training and test sets")
        x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42)
        logger.info(f"Training data shape: {x_training_data.shape}, Test data shape: {x_test_data.shape}")
    except Exception as e:
        logger.error(f"Error splitting data for Random Forest: {str(e)}")
        raise

    # Create and train the Random Forest Classifier Model
    try:
        logger.info("Training Random Forest model")
        # There is class imbalance problem, hence using class_weight='balanced'
        clf = RandomForestClassifier(
            bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=3, max_features='auto', max_leaf_nodes=3,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=999, n_jobs=1,
            oob_score=False, random_state=42, verbose=0,
            warm_start=False
        )
        clf.fit(x_training_data, y_training_data)
        logger.info("Successfully trained Random Forest model")
        
        # Make predictions
        y_pred = clf.predict(x_test_data)
        logger.info("Generated predictions with Random Forest model")
    except Exception as e:
        logger.error(f"Error training Random Forest model: {str(e)}")
        raise

    # Measure performance
    try:
        logger.info("Evaluating Random Forest model performance")
        
        # Create and print confusion matrix
        confusion_mat = pd.crosstab(y_test_data, y_pred, rownames=['Actual'], colnames=['Predicted'])
        print(confusion_mat)
        
        # Calculate accuracy
        accuracy = metrics.accuracy_score(y_test_data, y_pred)
        print('Accuracy: ', accuracy)
        logger.info(f"Random Forest accuracy: {accuracy:.4f}")
    except Exception as e:
        logger.error(f"Error evaluating Random Forest performance: {str(e)}")
        print(f"Warning: Could not evaluate Random Forest performance: {str(e)}")
        # Continue execution

    # Plot ROC curve
    try:
        logger.info("Plotting ROC curve for Random Forest model")
        fpr, tpr, _ = metrics.roc_curve(y_test_data, y_pred)
        auc = metrics.roc_auc_score(y_test_data, y_pred)
        plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.savefig('random_forest_roc_curve.png')  # Save the figure before showing
        logger.info(f"ROC curve plot saved to random_forest_roc_curve.png")
        plt.show()
    except Exception as e:
        logger.error(f"Error plotting ROC curve for Random Forest: {str(e)}")
        print(f"Warning: Could not create ROC curve plot: {str(e)}")
        # Continue execution

    # Identify variables that have highest impact on the outcome
    try:
        logger.info("Calculating feature importances")
        featureImportances = pd.Series(clf.feature_importances_).sort_values(ascending=False)
        print(featureImportances)
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sn.barplot(x=round(featureImportances, 4), y=featureImportances.index)
        plt.xlabel('Features Importance')
        plt.savefig('random_forest_feature_importance.png')
        logger.info(f"Feature importance plot saved to random_forest_feature_importance.png")
        plt.show()
        
        # Identify top features
        top_features = featureImportances.head(6).index.tolist()
        logger.info(f"Top 6 most important features: {top_features}")
    except Exception as e:
        logger.error(f"Error analyzing feature importances: {str(e)}")
        print(f"Warning: Could not analyze feature importances: {str(e)}")
        # Continue execution with predefined top features if needed
        top_features = ['euribor3m', 'poutcome_success', 'cons_conf_idx', 'cons_price_idx', 'previous_flag', 'poutcome_nonexistent']
        logger.warning(f"Using predefined top features: {top_features}")

    # Recreate the model and measure performance with only the top 6 most important factors
    try:
        logger.info("Building optimized Random Forest model with top features")
        
        # Check if all top features exist in model_df
        important_features = ['euribor3m', 'poutcome_success', 'cons_conf_idx', 'cons_price_idx', 'previous_flag', 'poutcome_nonexistent']
        
        # Check if all important features exist in model_df
        missing_features = [col for col in important_features if col not in model_df.columns]
        if missing_features:
            logger.warning(f"Missing important features: {missing_features}")
            # Use available features only
            available_features = [col for col in important_features if col in model_df.columns]
            if not available_features:
                raise KeyError("No important features available")
            logger.info(f"Using available features: {available_features}")
            x_data = model_df[available_features]
        else:
            # Use all important features
            y_data = model_df['classifier']
            x_data = model_df[important_features]
            
        # Split the data
        x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42)
        
        # Create and train the optimized model
        clf = RandomForestClassifier(
            bootstrap=True, class_weight='balanced', criterion='gini',
            max_depth=4, max_features='auto', max_leaf_nodes=4,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=999, n_jobs=1,
            oob_score=False, random_state=42, verbose=0,
            warm_start=False
        )
        clf.fit(x_training_data, y_training_data)
        y_pred = clf.predict(x_test_data)
        logger.info("Successfully trained optimized Random Forest model")
        
        # Evaluate optimized model
        confusion_mat = pd.crosstab(y_test_data, y_pred, rownames=['Actual'], colnames=['Predicted'])
        print(confusion_mat)
        
        accuracy = metrics.accuracy_score(y_test_data, y_pred)
        print('Accuracy: ', accuracy)
        logger.info(f"Optimized Random Forest accuracy: {accuracy:.4f}")
        
        # Plot ROC curve for optimized model
        plt.figure()
        fpr, tpr, _ = metrics.roc_curve(y_test_data, y_pred)
        auc = metrics.roc_auc_score(y_test_data, y_pred)
        plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
        plt.legend(loc=4)
        plt.savefig('optimized_random_forest_roc_curve.png')
        logger.info(f"Optimized ROC curve plot saved to optimized_random_forest_roc_curve.png")
        plt.show()
        
        # Feature importances for optimized model
        featureImportances = pd.Series(clf.feature_importances_, index=x_data.columns).sort_values(ascending=False)
        print(featureImportances)
        
        plt.figure(figsize=(10, 6))
        sn.barplot(x=round(featureImportances, 4), y=featureImportances.index)
        plt.xlabel('Features Importance')
        plt.savefig('optimized_random_forest_feature_importance.png')
        logger.info(f"Optimized feature importance plot saved to optimized_random_forest_feature_importance.png")
        plt.show()
        
        # Calculate comprehensive performance metrics
        metrics_dict = {
            'Accuracy': accuracy_score(y_test_data, y_pred),
            'Balanced Accuracy': balanced_accuracy_score(y_test_data, y_pred),
            'Precision': precision_score(y_test_data, y_pred),
            'Recall': recall_score(y_test_data, y_pred),
            'F1 Score': f1_score(y_test_data, y_pred),
            'ROC AUC': roc_auc_score(y_test_data, y_pred)
        }
        
        # Log all metrics
        for metric_name, metric_value in metrics_dict.items():
            logger.info(f"Optimized Random Forest {metric_name}: {metric_value:.4f}")
            
        # Return metrics for reference (they will be printed to console)
        accuracy_score(y_test_data, y_pred)
        balanced_accuracy_score(y_test_data, y_pred)
        precision_score(y_test_data, y_pred)
        recall_score(y_test_data, y_pred)
        f1_score(y_test_data, y_pred)
        roc_auc_score(y_test_data, y_pred)
        
    except KeyError as e:
        logger.error(f"Key error in optimized model: {str(e)}")
        print(f"Warning: Could not build optimized model due to missing features: {str(e)}")
    except Exception as e:
        logger.error(f"Error building optimized Random Forest model: {str(e)}")
        print(f"Warning: Could not build optimized Random Forest model: {str(e)}")
    
    logger.info("Random Forest modeling completed successfully")
    
except Exception as e:
    logger.error(f"Error in Random Forest modeling: {str(e)}")
    print(f"Error in Random Forest modeling: {str(e)}")

# Script completed
logger.info("Bank telemarketing analysis script completed")