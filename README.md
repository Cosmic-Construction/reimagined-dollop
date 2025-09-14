# XGBoost Training Pipeline

This repository contains an Apache Airflow DAG for training and evaluating XGBoost models using Amazon SageMaker.

## Overview

The `xgboost_training_pipeline` DAG implements a complete machine learning pipeline with the following tasks:

1. **Data Loading** - Loads or generates training data
2. **Data Preprocessing** - Cleans and prepares data for training  
3. **Model Training** - Trains an XGBoost model using SageMaker
4. **Model Evaluation** - Evaluates the trained model and generates metrics

## Features

- **Scalable Training**: Uses Amazon SageMaker for distributed XGBoost training
- **Automated Pipeline**: Complete ML workflow from data to evaluation
- **Error Handling**: Built-in retries and error handling
- **Configurable**: Easy to customize hyperparameters and resources
- **Monitoring**: Comprehensive logging and metric tracking

## Requirements

- Apache Airflow 2.5+
- AWS account with SageMaker access
- Python 3.8+

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure AWS credentials and SageMaker execution role

3. Update the DAG configuration in `dags/xgboost_training_pipeline.py`:
   - Replace `YOUR_ACCOUNT_ID` with your AWS account ID
   - Update S3 bucket paths
   - Adjust hyperparameters as needed

4. Deploy to your Airflow environment

## Configuration

### Key Parameters

- **Instance Type**: `ml.m5.large` (configurable)
- **Training Algorithm**: XGBoost 1.5-1
- **Default Hyperparameters**:
  - objective: binary:logistic
  - eval_metric: auc
  - num_round: 100
  - max_depth: 6
  - eta: 0.1

### DAG Settings

- **Schedule**: Manual trigger (schedule_interval=None)
- **Retries**: 2 attempts with 5-minute delay
- **Tags**: ml, xgboost, sagemaker

## Usage

1. Trigger the DAG from Airflow UI or CLI
2. Monitor progress through Airflow logs
3. Review model metrics in the evaluation task output

## Production Considerations

- Replace simulation tasks with actual SageMaker operators
- Implement proper S3 data management
- Add model versioning and artifact storage
- Set up monitoring and alerting
- Configure proper IAM roles and security

## Files

- `dags/xgboost_training_pipeline.py` - Main DAG definition
- `requirements.txt` - Python dependencies
- `README.md` - This documentation