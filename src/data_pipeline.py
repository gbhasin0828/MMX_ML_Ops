from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as spark_round
import os

# ✅ Initialize Spark Session
spark = SparkSession.builder.appName("DataTransformation").getOrCreate()

# ✅ Define Correct Paths for GitHub Actions
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # ✅ MMX_ML_Ops directory
input_path = os.path.join(BASE_DIR, "data", "data_new.csv")
output_path = os.path.join(BASE_DIR, "data", "data_training.parquet")

def transform_data():
    print(f"🚀 Reading data from: {input_path}")

    # ✅ Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Input file {input_path} not found!")
        return
    
    # ✅ Load CSV Data
    df = spark.read.csv(input_path, header=True, inferSchema=True)

    # ✅ Data Cleaning & Feature Engineering
    df_filtered = df.dropna()
    df_filtered = df_filtered.filter(col("sales") >= 75000000)
    df_filtered = df_filtered.withColumn("sales", spark_round(col("sales"), 0))
    df_filtered = df_filtered.withColumn("mdsp_sem_so", col("mdsp_sem") + col("mdsp_so"))

    # ✅ Ensure Output Directory Exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Save Transformed Data
    df_filtered.write.parquet(output_path, mode="overwrite")

    print(f"✅ Data Transformation Complete. Output Saved at: {output_path}")

# ✅ Define DAG for Airflow
default_args = {
    "owner": "airflow",
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
    task_id="transform_data",
    python_callable=transform_data,
    dag=dag,
)

transform_task
