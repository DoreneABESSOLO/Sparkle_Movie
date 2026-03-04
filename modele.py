from pyspark.ml.recommendation import ALS

def split_data(df, train_ratio=0.8):
    return df.randomSplit([train_ratio, 1 - train_ratio])


def train_als_model(train_df,
                    rank=10,
                    maxIter=10,
                    regParam=0.1):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=rank,
        maxIter=maxIter,
        regParam=regParam,
        coldStartStrategy="drop"
    )
    model = als.fit(train_df)
    return model
