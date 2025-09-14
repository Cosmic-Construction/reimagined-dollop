"""
XGBoost Training Pipeline DAG

This DAG trains and evaluates an XGBoost model using Amazon SageMaker.
It includes tasks for data loading, preprocessing, training, and evaluation.
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow.decorators import dag, task
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable

import pandas as pd
import numpy as np
import boto3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json


@dag(
    dag_id="xgboost_training_pipeline",
    default_args={"owner": "airflow", "retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["ml", "xgboost", "sagemaker"],
    description="A DAG to train and evaluate an XGBoost model using SageMaker.",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
)
def xgboost_training_pipeline():
    """
    DAG to train and evaluate an XGBoost model using SageMaker.
    Includes tasks for loading data, preprocessing, training, and evaluation.
    """

    @task
    def load_data() -> Dict[str, str]:
        """
        Load data from S3 or generate sample data for training.
        
        Returns:
            Dict containing data location information
        """
        # For demo purposes, we'll create sample data
        # In production, this would typically load from S3, database, or data lake
        
        # Generate sample classification dataset
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        # Generate target with some correlation to features
        y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.1) > 0
        y = y.astype(int)
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['target'] = y
        
        # Save to temporary location (in production, save to S3)
        data_path = '/tmp/sample_data.csv'
        df.to_csv(data_path, index=False)
        
        print(f"Generated {len(df)} samples with {n_features} features")
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        return {
            'data_path': data_path,
            'n_samples': n_samples,
            'n_features': n_features
        }

    @task
    def preprocess_data(data_info: Dict[str, str]) -> Dict[str, str]:
        """
        Preprocess the data for XGBoost training.
        
        Args:
            data_info: Dictionary containing data information from load_data task
            
        Returns:
            Dict containing preprocessed data paths
        """
        # Load the data
        df = pd.read_csv(data_info['data_path'])
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols]
        y = df['target']
        
        # Split data into train/validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply scaling (optional for XGBoost, but can help)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert back to DataFrames
        X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
        X_val_df = pd.DataFrame(X_val_scaled, columns=feature_cols)
        
        # For SageMaker XGBoost, we need CSV format with target as first column
        train_df = pd.concat([y_train.reset_index(drop=True), X_train_df], axis=1)
        val_df = pd.concat([y_val.reset_index(drop=True), X_val_df], axis=1)
        
        # Save preprocessed data
        train_path = '/tmp/train_data.csv'
        val_path = '/tmp/validation_data.csv'
        
        train_df.to_csv(train_path, index=False, header=False)
        val_df.to_csv(val_path, index=False, header=False)
        
        print(f"Training set shape: {train_df.shape}")
        print(f"Validation set shape: {val_df.shape}")
        
        return {
            'train_path': train_path,
            'validation_path': val_path,
            'train_samples': len(train_df),
            'val_samples': len(val_df)
        }

    @task
    def prepare_training_job(preprocessed_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Prepare SageMaker training job configuration.
        
        Args:
            preprocessed_data: Dictionary containing preprocessed data paths
            
        Returns:
            Dict containing training job configuration
        """
        # In production, you would upload data to S3 here
        # For this example, we'll prepare the training configuration
        
        # SageMaker XGBoost container URI (this would be region-specific)
        # Using built-in XGBoost algorithm container
        training_image = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        
        # Training job configuration
        training_config = {
            'job_name': f'xgboost-training-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            'algorithm_specification': {
                'TrainingImage': training_image,
                'TrainingInputMode': 'File'
            },
            'role_arn': 'arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerRole',  # Replace with actual role
            'input_data_config': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': 's3://your-bucket/training-data/',  # Replace with actual S3 path
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                },
                {
                    'ChannelName': 'validation',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': 's3://your-bucket/validation-data/',  # Replace with actual S3 path
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'text/csv',
                    'CompressionType': 'None'
                }
            ],
            'output_data_config': {
                'S3OutputPath': 's3://your-bucket/model-output/'  # Replace with actual S3 path
            },
            'resource_config': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            'stopping_condition': {
                'MaxRuntimeInSeconds': 3600
            },
            'hyperparameters': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'num_round': '100',
                'max_depth': '6',
                'eta': '0.1',
                'subsample': '0.8',
                'colsample_bytree': '0.8',
                'min_child_weight': '1',
                'silent': '1'
            }
        }
        
        print(f"Prepared training job: {training_config['job_name']}")
        print(f"Training samples: {preprocessed_data['train_samples']}")
        print(f"Validation samples: {preprocessed_data['val_samples']}")
        
        return training_config

    @task
    def simulate_training(training_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Simulate SageMaker training job (since we don't have actual AWS credentials).
        In production, this would be replaced by SageMakerTrainingOperator.
        
        Args:
            training_config: Training job configuration
            
        Returns:
            Dict containing training job results
        """
        import time
        import random
        
        job_name = training_config['job_name']
        
        print(f"Starting simulated training job: {job_name}")
        print("Training configuration:")
        print(f"  - Instance type: {training_config['resource_config']['InstanceType']}")
        print(f"  - Hyperparameters: {training_config['hyperparameters']}")
        
        # Simulate training time
        time.sleep(5)
        
        # Simulate training metrics
        final_auc = round(random.uniform(0.85, 0.95), 4)
        final_accuracy = round(random.uniform(0.80, 0.90), 4)
        
        training_results = {
            'job_name': job_name,
            'training_job_status': 'Completed',
            'model_artifacts_s3_path': f's3://your-bucket/model-output/{job_name}/output/model.tar.gz',
            'final_metric_auc': str(final_auc),
            'final_metric_accuracy': str(final_accuracy),
            'training_time_seconds': '300'
        }
        
        print(f"Training completed successfully!")
        print(f"Final AUC: {final_auc}")
        print(f"Final Accuracy: {final_accuracy}")
        print(f"Model artifacts: {training_results['model_artifacts_s3_path']}")
        
        return training_results

    @task
    def evaluate_model(training_results: Dict[str, str]) -> Dict[str, str]:
        """
        Evaluate the trained model and generate evaluation metrics.
        
        Args:
            training_results: Results from the training job
            
        Returns:
            Dict containing evaluation results
        """
        job_name = training_results['job_name']
        
        print(f"Evaluating model from job: {job_name}")
        
        # In production, you would:
        # 1. Download the model artifacts from S3
        # 2. Load test data
        # 3. Generate predictions
        # 4. Calculate various evaluation metrics
        
        # For this simulation, we'll create sample evaluation metrics
        evaluation_metrics = {
            'auc_score': training_results['final_metric_auc'],
            'accuracy': training_results['final_metric_accuracy'],
            'precision': '0.87',
            'recall': '0.84',
            'f1_score': '0.85',
            'confusion_matrix': '[[425, 75], [80, 420]]',
            'model_path': training_results['model_artifacts_s3_path']
        }
        
        print("Model Evaluation Results:")
        print(f"  - AUC Score: {evaluation_metrics['auc_score']}")
        print(f"  - Accuracy: {evaluation_metrics['accuracy']}")
        print(f"  - Precision: {evaluation_metrics['precision']}")
        print(f"  - Recall: {evaluation_metrics['recall']}")
        print(f"  - F1 Score: {evaluation_metrics['f1_score']}")
        
        # You could add logic here to determine if model meets quality thresholds
        auc_threshold = 0.8
        if float(evaluation_metrics['auc_score']) >= auc_threshold:
            print(f"✓ Model quality acceptable (AUC >= {auc_threshold})")
            evaluation_metrics['model_approved'] = 'true'
        else:
            print(f"✗ Model quality below threshold (AUC < {auc_threshold})")
            evaluation_metrics['model_approved'] = 'false'
            
        return evaluation_metrics

    # Define task dependencies
    data = load_data()
    preprocessed = preprocess_data(data)
    training_config = prepare_training_job(preprocessed)
    training_results = simulate_training(training_config)
    evaluation_results = evaluate_model(training_results)
    
    # Return final results
    return evaluation_results


# Create the DAG instance
xgboost_dag = xgboost_training_pipeline()