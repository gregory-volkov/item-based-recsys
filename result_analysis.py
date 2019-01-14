from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col
from pyspark import SQLContext
from pyspark import SparkContext
from metrics import *
import os

ndcg_k = 10

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python3"

SparkContext.setSystemProperty('spark.executor.memory', '5g')
sc = SparkContext("local", "Million songs")
sc.setLogLevel("WARN")


schema = StructType([
    StructField("Sid", IntegerType()),
    StructField("Uid", IntegerType()),
    StructField("Audit_n", IntegerType())
])

schema_user_ratings = StructType([
    StructField("Sid", IntegerType()),
    StructField("Audit_n", IntegerType())
])

schema_predicted = StructType([
    StructField("Uid", IntegerType()),
    StructField("Sid", IntegerType()),
    StructField("Predicted", DoubleType())
])

schema_metrics = StructType([
    StructField("MSE", DoubleType()),
    StructField("RMSE", DoubleType()),
    StructField("NDCG", DoubleType()),
    StructField("Gini", DoubleType())
])

sqlContext = SQLContext(sc)

result_df = (sqlContext
                .read
                .format("csv")
                .schema(schema_predicted)
                .option("delimiter", ',')
                .option("header", True)
                .load("predicted.csv"))


df = (sqlContext
        .read
        .format("csv")
        .schema(schema)
        .option("delimiter", ',')
        .option("header", True)
        .load("indexed_triplets.csv"))

metrics_df = sqlContext.createDataFrame(sc.emptyRDD(), schema=schema_metrics)

users_df = result_df.select("Uid").distinct().collect()
users = [x.Uid for x in users_df]

joined = df.alias("true")\
            .join(
            result_df.alias("predicted"), (col("true.Uid") == col("predicted.Uid")) &
                                          (col("true.Sid") == col("predicted.Sid")),
            how='inner'
            ).select("true.Uid", "true.Sid", "true.Audit_n", "predicted.Predicted")

for user in users:
    df_by_user = joined.filter(joined.Uid == user).select("Predicted", "Audit_n")
    predicted_v = np.array(df_by_user.select("Predicted").collect()).flatten()
    true_v = np.array(df_by_user.select("Audit_n").collect()).flatten()
    predicted_ids = np.argsort(predicted_v)
    metrics = tuple(map(float, [
                        mse(true_v, predicted_v),
                        rmse(true_v, predicted_v),
                        ndcg_score(true_v[predicted_ids], k=ndcg_k),
                        gini(predicted_v)
                        ]))

    temp_df = sqlContext.createDataFrame([metrics], schema=schema_metrics)
    temp_df.show()
    metrics_df = metrics_df.union(temp_df)

(metrics_df.repartition(1).write
    .format('com.databricks.spark.csv')
    .save("metrics", header='true', sep=','))
