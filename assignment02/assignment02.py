import pyspark

sc = pyspark.SparkContext('local[*]')

movieGenreRDD = sc.textFile('../Docs/movies.dat')
movieRatingsRDD = sc.textFile('../Docs/ratings.dat')
usersRDD = sc.textFile('../Docs/ratings.dat')



