include("utils.jl")

# slicer : [0,1] with 1 = 100%, default 80 %
function DataSlicer(path::String, boundary::Float64 = 0.8)
  println("...DataSlicer()")
  df = DataFrame(CSV.File(path))
  rows = size(df, 1)
  cols = size(df, 2)
  train, test = TrainTestSplit(df, boundary)
  names = propertynames(train)
  println("Features : ", cols, " | Patterns : ", rows, " | Boundary : ", boundary, " | Training : ", size(train, 1), " | Test : ", size(test, 1))
  println(names)

  return Dataset(names, cols, rows, boundary, Matrix(train), Matrix(test), train, test)
end

function normalisation(data::Dataset)
  return data
end


# MAPE function defination
function mape(performance_df)
  mape = mean(abs.(performance_df.error./performance_df.y_actual))
  return mape
end


function Multilinear_regression(data::Dataset)
  # train a linear regression model with GLM (Generalized Linear Model) package
  response = data.names[size(data.names, 1)]
  predictors = data.names[1:size(data.names, 1)-1]
  println(response)
  println(predictors)
  fm = Term(response) ~ sum(term.(predictors))
  linearRegressor = lm(fm, data.train_df)
  println(linearRegressor)

  # Model Prediction and Performance : Mean Absolute Percentage Error

#=
  # Prediction
  ypredicted_test = predict(linearRegressor, test)
  ypredicted_train = predict(linearRegressor, train)

  # Test Performance DataFrame (compute squared error)
  performance_testdf = DataFrame(y_actual = test[!,:Life_expectancy], y_predicted = ypredicted_test)
  performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
  performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

  # Train Performance DataFrame (compute squared error)
performance_traindf = DataFrame(y_actual = train[!,:Life_expectancy], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error ;

# Test Error
println("Mean Absolute test error: ",mean(abs.(performance_testdf.error)), "\n")
println("Mean Aboslute Percentage test error: ",mape(performance_testdf), "\n")

=#
end

data = DataSlicer("dataset/A1-turbine.txt", 0.85)
Multilinear_regression(data)
