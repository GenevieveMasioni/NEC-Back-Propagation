include("utils.jl")
include("NeuralNet.jl")

struct Dataset
  features::Int64                # number of features
  patterns::Int64                # number of patterns
  boundary::Float64              # percentage of patterns used for training-validation set [0,1]
  train::DataFrame            # training-validation set
  test::DataFrame                # test set
end

# slicer : [0,1] with 1 = 100%, default 80 %
function DataSlicer(path::String, boundary::Float64 = 0.8)
  println("...DataSlicer()")
  df = DataFrame(CSV.File(path))
  rows = size(df, 1)
  cols = size(df, 2)
  train, test = TrainTestSplit(df, boundary)
  println("Features : ", cols, " | Patterns : ", rows, " | Boundary : ", boundary, " | Training : ", size(train, 1), " | Test : ", size(test, 1))
  return Dataset(cols, rows, boundary, train, test)
end


function Multilinear_regression(data::Dataset)
  #= train a linear regression model with GLM (Generalized Linear Model) package
  fm = @formula(fall ~ power_of_hydroelectrical_turbine)
  linearRegressor = lm(fm, data.train)
  # Model Prediction and Performance : Mean Absolute Percentage Error
  =#

  println("solve using llsq")
  X = Matrix(data.train)
  y = data.test
  a, b = llsq(X, y)
  println(a)
  println(b)

  println("do prediction")
  yp = X * a + b
  println(yp)

  println("measure the error")
  rmse = sqrt(mean(abs2(y - yp)))
  print("rmse = $rmse")

end

data = DataSlicer("dataset/A1-turbine.txt", 0.85)
Multilinear_regression(data)
