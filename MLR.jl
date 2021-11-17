include("utils.jl")

# Mean Absolute Percentage Error
function mape(performance)
  mape = mean(abs.(performance.error./performance.y_actual))
  return mape
end

function Multilinear_regression(data::Dataset)
  println("...Multilinear_regression()")
  # train a linear regression model with GLM (Generalized Linear Model) package
  response = data.names[size(data.names, 1)]
  predictors = data.names[1:size(data.names, 1)-1]
  #println(response)
  #println(predictors)
  fm = Term(response) ~ sum(term.(predictors))
  linearRegressor = lm(fm, data.train_df)
  println(linearRegressor)

  # R Square value of the model
  println("R-Square : ", r2(linearRegressor), "\n")

  # Model Prediction and Performance : Mean Absolute Percentage Error

  # Prediction
  prediction_test = predict(linearRegressor, data.test_df)
  prediction_train = predict(linearRegressor, data.train_df)

  # Training Performance DataFrame (compute squared error)
  performance_train = DataFrame(y_actual = data.train_df[!,response], y_predicted = prediction_train)
  performance_train.error = performance_train[!,:y_actual] - performance_train[!,:y_predicted]
  performance_train.error_sq = performance_train.error.*performance_train.error

  # Training Error
  println("Mean Absolute train error: ", mean(abs.(performance_train.error)), "\n")
  println("Mean Aboslute Percentage train error: ", mape(performance_train), "\n")

  # Test Performance DataFrame (compute squared error)
  performance_test = DataFrame(y_actual = data.test_df[!,response], y_predicted = prediction_test)
  performance_test.error = performance_test[!,:y_actual] - performance_test[!,:y_predicted]
  performance_test.error_sq = performance_test.error.*performance_test.error

  # Test Error
  println("Mean Absolute test error: ", mean(abs.(performance_test.error)), "\n")
  println("Mean Absolute Percentage test error: ", mape(performance_test), "\n")

  # Histogram of error to see if it's normally distributed on train and test datasets
  histogram(performance_train.error, bins = 50, title = "Train Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)
  histogram(performance_test.error, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)
end


