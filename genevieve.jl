include("utils.jl")

struct Dataset
  features::Int64                # number of features
  patterns::Int64                # number of patterns
  boundary::Float64              # percentage of patterns used for training-validation set [0,1]
  training::DataFrame            # training-validation set
  test::DataFrame                # test set
end

# slicer : [0,1] with 1 = 100%, default 80 %
function DataSlicer(path::String, boundary::Float64 = 0.8)
  println("...DataSlicer()")
  df = DataFrame(CSV.File(path))
  rows = size(df, 1)
  cols = size(df, 2)
  maxTraining = round(Int64, rows * boundary)
  trainingSet = first(df, maxTraining)
  testSet = last(df, rows - maxTraining)
  println("Features : ", df.features, " | Patterns : ", df.patterns, " | Boundary : ", df.boundary, " | Training : ", size(df.training, 1), " | Test : ", size(df.test, 1))
  return Dataset(cols, rows, boundary, trainingSet, testSet)
end


df = DataSlicer("dataset/A1-turbine.txt", 0.85)
