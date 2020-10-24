from pyspark import Row, SparkConf
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StructField

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


def f(x):
    d = {}
    x = parseMovieGenres(x)
    for i in range(len(x)):
        d[str(i)] = x[i]
    return d


def getSchema():
    genres = movieGenreRDD.map(lambda x: Row(**f(x))).map(lambda x: x[1:])
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
    for k in range(2, 20):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
        model = kmeans.fit(df.sample(False, 0.8, seed=32))
        predictions = model.transform(df)
        cost[k] = evaluator.evaluate(predictions)
    plotK(cost)


def run():
    schema = getSchema()
    movieGenreWithValuesRDD = movieGenreRDD.map(lambda x: Row(**f(x))).map(lambda x: fillInValues(schema, x))
    df = spark.createDataFrame(movieGenreWithValuesRDD.collect(), schema)
    dfVectorGenres = vectorizeFeatures(df)
    dfVectorGenres.show()
    getOptimalKPlot(dfVectorGenres)


run()
