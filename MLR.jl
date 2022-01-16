include("utils.jl")
include("back_propagation.jl")

# Mean Absolute Percentage Error
function mape(performance)
  mape = mean(abs.(performance.error./performance.y_actual))
  return mape
end

function MLR(data::Dataset)
  println("...MLR()")
  response = data.names[size(data.names, 1)]
  predictors = data.names[1:size(data.names, 1)-1]
  fm = Term(response) ~ sum(term.(predictors))
  linearRegressor = lm(fm, data.train_df)
  println(linearRegressor)

  # Prediction
  prediction_test = predict(linearRegressor, data.test_df)
  prediction_train = predict(linearRegressor, data.train_df)

  # Test Performance DataFrame (compute squared error)
  performance_test = DataFrame(y_actual = data.test_df[!,response], y_predicted = prediction_test)
  
  performance_test.error = performance_test[!,:y_actual] - performance_test[!,:y_predicted]
  performance_test.error_sq = performance_test.error.*performance_test.error

  # Test Error
  println("Mean Absolute test percentage error: ", mean(abs.(performance_test.error))*100, "%\n")

  # save results
  performance_test_csv = deepcopy(data.test_df)
  insertcols!(performance_test_csv, size(data.test_df,2)+1, :y_predicted => prediction_test[:])
  descale(performance_test_csv, data.rangesTest)
  s = size(performance_test_csv,2)
  CSV.write(string("./Results/MLR/_results_test.csv"), performance_test_csv[:,s-1:s])

  #Plots
  figureRPTe = scatter(performance_test_csv[:,s-1],performance_test_csv[:,s],title = "Predicted Vs Original Test", ylabel="Prediction", xlabel="Original")
  display(figureRPTe)
  #save Plots
  png(figureRPTe,string("Plots/MLR/figure_Real_Predict_Test.jpg"))

  return mean(abs.(performance_test.error))*100
end

function crossValidation(nn::NeuralNet, data::Dataset, nbFolds::Int64, η::Float64, α::Float64, epochs::Int64)
  println("...crossValidation()")
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

    rangesTrain = getRanges(train)
    rangesTest = getRanges(test)
    dataset = Dataset(data.names, data.features, data.patterns, data.boundary, Matrix(train), Matrix(test), train, test, rangesTrain, rangesTest)
    scale(train, rangesTrain)
    scale(test, rangesTest)
    error_bp += BP(nn, dataset, η, α, epochs)
    error_mlr += MLR(dataset)
  end
  # compute global error
  return (error_bp/nbFolds, error_mlr/nbFolds)
end
