
# ----- Import Necessary packages -------
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.shell import spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler


# Reading in data
flights_sdf = spark.read.csv('processed.csv', header=True, inferSchema=True)
flights_sdf.show()

# Creating vector assembler with desired feature columns
feature_columns = [
 'OP_CARRIER_FL_NUM',
 'CRS_DEP_TIME',
 'DEP_TIME',
 'DEP_DELAY',
 'TAXI_OUT',
 'WHEELS_OFF',
 'WHEELS_ON',
 'TAXI_IN',
 'CRS_ARR_TIME',
 'ARR_TIME',
 'CANCELLED',
 'DIVERTED',
 'CRS_ELAPSED_TIME',
 'ACTUAL_ELAPSED_TIME',
 'AIR_TIME',
 'DISTANCE',
 'CARRIER_DELAY',
 'WEATHER_DELAY',
 'NAS_DELAY',
 'SECURITY_DELAY',
 'LATE_AIRCRAFT_DELAY',
 'Unnamed: 27',
 'avg_delay',
 'FL_weekday',
 'OP_CARRIER_AA',
 'OP_CARRIER_AS',
 'OP_CARRIER_B6',
 'OP_CARRIER_DL',
 'OP_CARRIER_EV',
 'OP_CARRIER_F9',
 'OP_CARRIER_HA',
 'OP_CARRIER_NK',
 'OP_CARRIER_OO',
 'OP_CARRIER_UA',
 'OP_CARRIER_VX',
 'OP_CARRIER_WN']

# VectorAssembler used to multiple columns to single vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

data = assembler.transform(flights_sdf)
data = data.select(['features', 'ARR_DELAY'])

# Splitting into train and test datasets
train, test = data.randomSplit([0.8, 0.2])

# Fitting Linear Regression Model
lreg_v1 = LinearRegression(featuresCol="features", labelCol="ARR_DELAY")
lr_model_v1 = lreg_v1.fit(train)

# Predicting and Finding R2 and RMSE Values
predictions_v1 = lr_model_v1.transform(test)
eval_reg = RegressionEvaluator(labelCol="ARR_DELAY", metricName="r2")
test_result_v1 = lr_model_v1.evaluate(test)

print("R Squared (R2) on test data = %g" % eval_reg.evaluate(predictions_v1))
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result_v1.rootMeanSquaredError)
