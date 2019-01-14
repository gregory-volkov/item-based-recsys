from utils import get_user_ratings, get_sid_similarities
from pyspark.sql.functions import col
from pyspark.sql import functions as F


def predict(user, item, items_stat, rating_mat, sim_mat, sql_context, user_rating_schema, song_sim_schema):
    item_avg = items_stat.filter(items_stat.Sid == item).first().Mean

    users_ratings = get_user_ratings(sql_context, rating_mat, user, user_rating_schema)

    sims = get_sid_similarities(sql_context, sim_mat, item, song_sim_schema)

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

    return item_avg + nom / denom if denom else 0.
