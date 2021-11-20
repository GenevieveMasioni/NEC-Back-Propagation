include("utils.jl")
include("back_propagation.jl")

# Mean Absolute Percentage Error
function mape(performance)
  mape = mean(abs.(performance.error./performance.y_actual))
  return mape
end

function MLR(data::Dataset)
  println("...MLR()")
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
  println("Mean Absolute Percentage train error: ", mape(performance_train), "\n")

  # Test Performance DataFrame (compute squared error)
  performance_test = DataFrame(y_actual = data.test_df[!,response], y_predicted = prediction_test)
  performance_test.error = performance_test[!,:y_actual] - performance_test[!,:y_predicted]
  performance_test.error_sq = performance_test.error.*performance_test.error

  # Test Error
  println("Mean Absolute test error: ", mean(abs.(performance_test.error)), "\n")
  println("Mean Absolute Percentage test error: ", mape(performance_test), "\n")

  #= Histogram of error to see if it's normally distributed on train and test datasets
  histogram(performance_train.error, bins = 50, title = "Train Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)
  histogram(performance_test.error, bins = 50, title = "Test Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false)
  =#
  return mape(performance_test)
end

# repeat the training process n-folds times and find optimal parameters (architecture, learning rate, momemtum nnuomber of epochs)
function crossValidation(nn::NeuralNet, data::Dataset, nbFolds::Int64)
  println("...crossValidation()")
  η = 0.05
  α = 0.1
  #dataset = copy(data)
  error_bp = 0
  error_mlr = 0
  folds = []
  foldSize = round(Int64, size(data.train_df, 1) / nbFolds)
  println("Fold : ", nbFolds, " - Fold size : ", foldSize, " - Patterns : ", data.patterns)

  for i in 1:nbFolds
    start = (i-1) * foldSize + 1
    max = i * foldSize
    if(max > size(data.train_df,1))
      max = size(data.train_df,1)
    end
    println("start = ", start, " ; max = ", max)
    fold = copy(data.train_df[start:max, :])
    push!(folds, fold)
  end

  for i in 1:nbFolds
    # train with folds-1 subsets and validate with the last one
    test = folds[i]
    train = deepcopy(folds)
    deleteat!(train, i)
    train = vcat(select.(train, :)...)

    # compute error x
    rangesTrain = getRanges(train)
    rangesTest = getRanges(test)
    dataset = Dataset(data.names, data.features, data.patterns, data.boundary, Matrix(train), Matrix(test), train, test, rangesTrain, rangesTest)
    scale(train, rangesTrain)
    scale(test, rangesTest)
    error_bp += BP(nn, dataset, η, α)
    error_mlr += MLR(dataset)
  end
  # computer global error
  return (error_bp/nbFolds, error_mlr/nbFolds)
end


#=
Another function : compute perf
once best params found, train complete dataset with those params
and test with test dataset / Use final NN to predict over test dataset
Evaluate test error

function findBestParameters()
  # inputs : user inputs (epochs, folds, etc.)
  crossValidation() # run x times and store prediction errors and value of params
  # select params with lowest prediction error
  # training with complete training dataset
  # evaluation using test dataset => final error
end
=#
