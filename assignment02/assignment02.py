from pyspark import Row, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StructField
from pyspark.sql.functions import col, avg

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
    return optimalmodel.transform(df).drop(df.features)


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
    return ratingsdf.join(moviesDF, ["movieID"], how="right")

def run():
    schema = getMovieSchema()
    movieGenreWithValuesRDD = movieGenreRDD.map(lambda x: Row(**movieParser(x))).map(lambda x: fillInValues(schema, x))
    df = spark.createDataFrame(movieGenreWithValuesRDD.collect(), schema)
    dfVectorGenres = vectorizeFeatures(df)
    dfVectorGenres.show()
    getOptimalKPlot(dfVectorGenres)
    predictionsDF = getOptimalClustersDF(dfVectorGenres)  # optimal k appears to be 6
    joinedDF = joinRatingsAndMovieClusters(predictionsDF).sort("userID", "movieID")
    # trainDF, testDF = joinedDF.randomSplit([.8, .2])
    # trainDF = trainDF.sort("userID", "movieID")
    # testDF = testDF.sort("userID", "movieID")
    joinedAvgDF = joinedDF.groupBy("userID", "prediction").agg({"rating": "avg"})
    joinedAvgDF = joinedAvgDF.sort("userID", "prediction")
    joinedAvgDF.show()


run()
