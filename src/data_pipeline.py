
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round
import pandas as pd
import numpy as np
import os

spark = SparkSession.builder.appName("DataTransformation").getOrCreate()

# ✅ Define File Paths (Updated for Local System)
input_path = "/Users/gauravbhasin/Desktop/DevOps_and_MLOps/MMX_ML_Ops/data/data_new.csv"
output_path = "/Users/gauravbhasin/Desktop/DevOps_and_MLOps/MMX_ML_Ops/data/data_training.parquet"

def transform_data():
  df = spark.read.csv(input_path, header=True, inferSchema=True)

  df_filtered = df.dropna()
  df_filtered = df_filtered.filter(col("sales") >= 75000000)

  df_filtered = df_filtered.withColumn("sales", spark_round(col("sales"), 0))

  df_filtered = df_filtered.withColumn("mdsp_sem_so", col("mdsp_sem") + col("mdsp_so"))

  df_filtered.write.parquet(output_path, mode = "overwrite")

  print(f"✅ Data Transformation Complete. Output Saved at: {output_path}")

# ✅ Define DAG for Airflow
default_args = {
    "owner" : "airflow",
    "depends_on_past": False,
    "start_date": datetime(2024, 2, 20),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    "data_transformation_pipeline",
    default_args=default_args,
    description="Data transformation pipeline using PySpark",
    schedule_interval="@daily",
)

transform_task = PythonOperator(
    task_id = "transform_data",
    python_callable = transform_data,
    dag = dag,
)

transform_task
