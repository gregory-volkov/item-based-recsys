from time import time
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType, DoubleType, LongType
from pyspark.sql.functions import countDistinct, col
from pyspark.sql import functions as F
from pyspark import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.mllib.linalg.distributed import CoordinateMatrix
from pyspark.mllib.linalg import SparseMatrix, SparseVector
import pickle
import numpy as np
import os


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

print("filtering")
df = df.filter(df.Sid <= 1000)

print("LOADED")

#sub_dfs = df.randomSplit([1.0] * 10)

df.show()
print(df.count())

row = df.first()
user_test = row.Uid
song_test = row.Sid
df = df.filter(f"Uid != {user_test}" or f"Sid != {song_test}")
print("test", user_test, song_test)


def get_sid_similarities(sim_mat, sid):
    return sqlContext.createDataFrame(
        (sim_mat.entries
            .filter(lambda x: x.i == sid or x.j == sid)
            .map(lambda x: (x.i + x.j - sid, x.value))),
        schema=schema_song_sims
    )


def get_user_ratings(rating_mat, uid):
    return sqlContext.createDataFrame(
        rating_mat.rows.filter(lambda x: x.vector[uid] > 0).map(lambda x: (x.index, int(x.vector[uid]))),
        schema=schema_user_ratings
    )


def vector_by_sid(rating_mat, sid):
    return rating_mat.rows.filter(lambda x: x.index == sid).first()


def average_rating_by_user(rating_mat, uid):
    return rating_mat.rows.filter(lambda x:  x.vector[uid] > 0).map(lambda x: x.vector[uid]).mean()


rating_mat = CoordinateMatrix(
    df.rdd.map(tuple)
)

sim_mat = rating_mat.transpose().toIndexedRowMatrix().columnSimilarities()

rating_mat = rating_mat.toIndexedRowMatrix()

rating_mat.rows.map(lambda x: x.vector.toSparse())

item_to_stat = rating_mat.rows.map(lambda x: (x.index, float(x.vector.values.mean()), float(x.vector.values.std())))
items_stat = sqlContext.createDataFrame(item_to_stat, schema=schema_items_stat)

t = time()

item_avg = items_stat.filter(items_stat.Sid == song_test).first().Mean

users_ratings = get_user_ratings(rating_mat, user_test)

sims = get_sid_similarities(sim_mat, song_test)

formula_data = users_ratings.alias("ratings").join(
    (sims
        .alias("sims")
        .join(items_stat.alias("stats"), col("sims.Sid") == col("stats.Sid"), how='left')
        .select("sims.Sid", "sims.Cos", "stats.Mean", "stats.Std")
     ).alias("sims_stats"),
    col("sims_stats.Sid") == col("ratings.Sid"),
    how="left"
).select("ratings.Sid", "ratings.Audit_n", "sims_stats.Mean", "sims_stats.Std", "sims_stats.Cos")

formula_data = (formula_data
                .withColumn("NomItem", col("Cos") * (col("Audit_n") - col("Mean"))))


nom = formula_data.select(F.sum(col("NomItem"))).collect()[0][0]
denom = formula_data.select(F.sum(col("Cos"))).collect()[0][0]

formula_data.show()

print(item_avg + nom / denom)

print(time() - t)
