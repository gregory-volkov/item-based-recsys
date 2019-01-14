from predictor import predict
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import col
from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import Row
from utils import get_dfs
import os

min_sid = 100
max_sid = 200
min_uid = 1000
max_uid = 10000

parts_n = 5


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

schema_items_stat = StructType([
    StructField("Sid", IntegerType()),
    StructField("Mean", DoubleType()),
    StructField("Std", DoubleType())
])

schema_user_ratings = StructType([
    StructField("Sid", IntegerType()),
    StructField("Audit_n", IntegerType())
])

schema_song_sims = StructType([
    StructField("Sid", IntegerType()),
    StructField("Cos", DoubleType())
])

sqlContext = SQLContext(sc)

df = (sqlContext
        .read
        .format("csv")
        .schema(schema)
        .option("delimiter", ',')
        .option("header", True)
        .load("indexed_triplets.csv"))

df = df.filter((min_sid <= col("Sid")) & (col("Sid") <= max_sid) &
               (min_uid <= col("Uid")) & (col("Uid") <= max_uid))

sub_dfs = df.randomSplit([1.0] * parts_n)

predicted_df = sqlContext.createDataFrame(sc.emptyRDD(), schema)

for i in range(parts_n):
    print(f"Validation of the {i} sub dataframe")
    test_df = sub_dfs[i]
    train_df = sqlContext.createDataFrame(sc.emptyRDD(), schema)

    for i_df in range(parts_n):
        if i_df != i:
            train_df = train_df.union(sub_dfs[i_df])

    test_sample = test_df.first()
    user = test_sample.Uid

    rating_mat, items_stat, sim_mat = get_dfs(train_df, sqlContext, schema_items_stat)

    for item in range(min_sid, max_sid + 1):
        predicted = predict(user, item, items_stat, rating_mat, sim_mat, sqlContext, schema_user_ratings, schema_song_sims)

        new_df = sc.parallelize([Row(Sid=item, Uid=user, Audit_n=predicted)]).toDF()
        predicted_df = predicted_df.union(new_df)

(predicted_df.repartition(1).write
    .format('com.databricks.spark.csv')
    .save("predicted", header='true', sep=','))
