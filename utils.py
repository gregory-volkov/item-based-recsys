from pyspark.mllib.linalg.distributed import CoordinateMatrix


def get_dfs(df, sql_context, items_stat_schema):
    rating_mat = CoordinateMatrix(
        df.rdd.map(tuple)
    )

    sim_mat = rating_mat.transpose().toIndexedRowMatrix().columnSimilarities()

    rating_mat = rating_mat.toIndexedRowMatrix()

    rating_mat.rows.map(lambda x: x.vector.toSparse())

    item_to_stat = rating_mat.rows.map(lambda x: (x.index, float(x.vector.values.mean()), float(x.vector.values.std())))
    items_stat = sql_context.createDataFrame(item_to_stat, schema=items_stat_schema)

    return rating_mat, items_stat, sim_mat


def get_sid_similarities(sql_context, sim_mat, sid, schema):
    return sql_context.createDataFrame(
        (sim_mat.entries
            .filter(lambda x: x.i == sid or x.j == sid)
            .map(lambda x: (x.i + x.j - sid, x.value))),
        schema=schema
    )


def get_user_ratings(sql_context, rating_mat, uid, schema):
    return sql_context.createDataFrame(
        rating_mat.rows.filter(lambda x: x.vector[uid] > 0).map(lambda x: (x.index, int(x.vector[uid]))),
        schema=schema
    )
