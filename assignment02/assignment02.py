from pyspark import Row, SparkConf
from pyspark.sql import SparkSession
import pyspark
from pyspark.sql.types import StructType, IntegerType, StructField

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


def fillInValues(schema: StructType, line: list):
    columnNames = schema.fieldNames()
    valuesIncluded = [""] * (len(columnNames) + 1)
    count = 0
    for genre in line:
        if count == 0:
            valuesIncluded[0] = genre
        elif genre in columnNames:
            # plus 1 is because the other list starts with movieID
            position = columnNames.index(genre) + 1
            valuesIncluded[position] = "1"
        count += 1
    for index in range(0, len(valuesIncluded)):
        if valuesIncluded[index] == "":
            valuesIncluded[index] = "0"
    return valuesIncluded


def run():
    movieGenreWithValuesRDD = movieGenreRDD
    genres = movieGenreRDD.map(lambda x: Row(**f(x))).map(lambda x: x[1:])
    genres = list(set(genres.flatMap(lambda x: x).collect()))
    genres.sort()
    schema = StructType([StructField(name, IntegerType()) for name in genres])
    print(schema.fieldNames())
    movieGenreWithValuesRDD = movieGenreWithValuesRDD.map(lambda x: Row(**f(x))).map(lambda x: fillInValues(schema, x))
    print(movieGenreWithValuesRDD.collect()[0])



run()
