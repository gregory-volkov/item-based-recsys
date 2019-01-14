from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import StringType, IntegerType
from pyspark import SQLContext
from pyspark.ml.feature import StringIndexer
from pyspark import SparkContext
from pyspark.ml import Pipeline
import os

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python3"

SparkContext.setSystemProperty('spark.executor.memory', '6g')
sc = SparkContext("local", "Million songs")
sc.setLogLevel("WARN")

schema = StructType([
    StructField("UserName", StringType()),
    StructField("SongName", StringType()),
    StructField("Audit_n", IntegerType())
])

sqlContext = SQLContext(sc)

df = (sqlContext
        .read
        .format("csv")
        .schema(schema)
        .option("delimiter", '\t')
        .option("header", False)
        .load("train_triplets.txt")
)

index_columns = [
    ("UserName", "Uid"),
    ("SongName", "Sid")
]

indexers = [StringIndexer(inputCol=inputCol, outputCol=outputCol) for inputCol, outputCol in index_columns]

pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

df = df.select("Sid", "Uid", "Audit_n")

df = df.withColumn("Sid", df["Sid"].cast(IntegerType()))
df = df.withColumn("Uid", df["Uid"].cast(IntegerType()))

(df.repartition(1).write
    .format('com.databricks.spark.csv')
    .save("indexed_triplets", header='true'))
