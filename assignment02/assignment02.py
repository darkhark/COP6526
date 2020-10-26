from pyspark import Row, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StructField
import pyspark.sql.functions as psf
import pyspark
import numpy as np
import matplotlib.pyplot as plt
# need to install tkinter to see plots using sudo apt-get install python3-tk

conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
sc = pyspark.SparkContext('local[*]')
spark = SparkSession(sc)

movieGenreRDD = sc.textFile('../Docs/movies.dat')
movieRatingsRDD = sc.textFile('../Docs/ratings.dat')


def parseMovieGenres(genreLine: str):
    """
    Goes through the movies.dat rdd to separate the different fields into their own "cell" in the list.

    :param genreLine: A single line from an RDD in movieGenreRDD
    :return: A parsed line where the fields are movieID, movieTitle, list(genres)
    """
    fields = genreLine.strip().split("::")
    fields[1] = fields[2]
    del fields[2]
    fields[1] = fields[1].strip().split("|")
    completeList = []
    for field in fields:
        if isinstance(field, list):
            for innerField in field:
                completeList.append(innerField)
        else:
            completeList.append(field)
    return completeList


def movieParser(x):
    d = {}
    x = parseMovieGenres(x)
    for i in range(len(x)):
        d[str(i)] = x[i]
    return d


def getMovieSchema():
    genres = movieGenreRDD.map(lambda x: Row(**movieParser(x))).map(lambda x: x[1:])
    genres = list(set(genres.flatMap(lambda x: x).collect()))
    genres.sort()
    genres = ["movieID"] + genres
    schema = StructType([StructField(name, IntegerType()) for name in genres])
    return schema


def fillInValues(schema: StructType, line: list):
    columnNames = schema.fieldNames()[1:]
    valuesIncluded = [0] * (len(columnNames) + 1)
    count = 0
    for genre in line:
        if count == 0:
            valuesIncluded[0] = int(genre)
        elif genre in columnNames:
            # plus 1 is because the other list starts with movieID
            position = columnNames.index(genre) + 1
            valuesIncluded[position] = 1
        count += 1
    for index in range(0, len(valuesIncluded)):
        if valuesIncluded[index] == "":
            valuesIncluded[index] = 0
    return valuesIncluded


def vectorizeFeatures(df):
    vectorizer = VectorAssembler(inputCols=df.columns[1:], outputCol="features")
    dfVectorGenres = vectorizer.transform(df).select("movieID", "features")
    return dfVectorGenres


def plotK(cost):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(range(2, 20), cost[2:20])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.show()


def getOptimalKPlot(df):
    cost = np.zeros(20)
    evaluator = ClusteringEvaluator()
    sample = df.sample(False, 0.8, seed=32)
    for k in range(2, 20):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
        model = kmeans.fit(sample)
        predictions = model.transform(df)
        cost[k] = evaluator.evaluate(predictions)
    plotK(cost)


def getOptimalClustersDF(df):
    optimalkmeans = KMeans().setK(6).setSeed(1).setFeaturesCol("features")
    optimalmodel = optimalkmeans.fit(df)
    return optimalmodel.transform(df).drop(df.features), optimalmodel


def parseMovieRatings(ratingLine: str):
    fields = ratingLine.strip().split("::")
    del fields[-1]
    return fields


def ratingsParser(x):
    d = {}
    x = parseMovieRatings(x)
    for i in range(len(x)):
        d[str(i)] = int(x[i])
    return d


def getRatingsSchema():
    columns = ["userID", "movieID", "rating"]
    return StructType([StructField(name, IntegerType()) for name in columns])


def getRatingsDF():
    rdd = movieRatingsRDD.map(lambda x: Row(**ratingsParser(x)))
    return spark.createDataFrame(rdd.collect(), getRatingsSchema())


def joinRatingsAndMovieClusters(moviesDF):
    ratingsdf = getRatingsDF()
    return ratingsdf.join(moviesDF, ["movieID"], how="left")


# def getAvgMissingRatingsDF(joindf, movieDF):
#     ratingDF = getRatingsDF()
#     movies = movieDF.select("movieID").distinct()
#     movies.show()
#     users = ratingDF.select("userID").distinct()
#     users.show()
#     movieForEachUserDF = movies.crossJoin(users)
#     movieForEachUserDF.show()
#     ratingAndAvgDF = movieForEachUserDF.join(
#         joindf,
#         [movieForEachUserDF.movieID, joindf.userID],
#         how="left"
#     ).sort("userID", "movieID")
#     ratingAndAvgDF.show()
#     # return ratingAndAvgDF

def getRMSE(df):
    rmseDF = df.drop(df.userID).drop(df.prediction).drop(df.movieID)
    rmse = rmseDF.withColumn("squarederror",
                             psf.pow(psf.col("avg(rating)") - psf.col("rating"),
                                     psf.lit(2))
                             )\
        .agg(psf.avg(psf.col("squarederror")).alias("mse"))\
        .withColumn("rmse", psf.sqrt(psf.col("mse")))
    return rmse.collect()


def runTask1():
    schema = getMovieSchema()
    movieGenreWithValuesRDD = movieGenreRDD.map(lambda x: Row(**movieParser(x))).map(lambda x: fillInValues(schema, x))
    df = spark.createDataFrame(movieGenreWithValuesRDD.collect(), schema)
    dfVectorGenres = vectorizeFeatures(df)
    dfVectorGenres.show()
    predictedClustersDF, optimalKModel = getOptimalClustersDF(dfVectorGenres)  # optimal k appears to be 6
    joinedDF = joinRatingsAndMovieClusters(predictedClustersDF).sort("userID", "movieID")
    # trainDF, testDF = joinedDF.randomSplit([.8, .2])
    # trainDF = trainDF.sort("userID", "movieID")
    # testDF = testDF.sort("userID", "movieID")
    joinedAvgDF = joinedDF.groupBy("userID", "prediction").agg({"rating": "avg"})
    joinedAvgDF = joinedAvgDF.join(joinedDF, ["userID", "prediction"], how="left").sort("userID", "movieID", "prediction")
    joinedAvgDF.show()
    print(getRMSE(joinedAvgDF))


#Create a function to find the best hyper parameters for tuning
def runTask2(ranks, regParams, numIters):
    minimumError = 10000
    bestRank = 0
    bestReg = 0
    bestIter = 0
    bestModel = None
    ratingsDF = getRatingsDF()
    ratingsTrain, ratingsTest = ratingsDF.randomSplit([.8, .2], seed=32)
    for rank in ranks:
        for item in regParams:
            for maxIter in numIters:
                als = ALS(userCol="userID", itemCol="movieID", ratingCol="rating",
                          nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")\
                    .setRank(rank).setRegParam(item).setMaxIter(maxIter)
                alsModel = als.fit(ratingsTrain)
                preds = alsModel.transform(ratingsTest)
                evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
                rmse = evaluator.evaluate(preds)
                print('{} latent factors,  regs = {}, maxIters = {}: validation rmse is {}'
                      .format(rank, item, maxIter, rmse))
                if rmse < minimumError:
                    minimumError=rmse
                    bestRank = rank
                    bestReg = item
                    bestModel = alsModel
                    bestIter = maxIter
    print('\nThe best model has {} latent factors, reg = {}, and iterations {}'.format(bestRank, bestReg, bestIter))
    return bestModel


runTask1()
runTask2([5, 10, 25, 50, 100], [1, .5, .1, .01], [5, 10, 15])
