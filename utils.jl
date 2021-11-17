# Import Packages
using Pkg  # Package to install new packages

# Install packages
#=
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("GLM")
Pkg.add("Plots")
Pkg.add("Lathe")
#Pkg.add("MultivariateStats")
Pkg.add("StatsModels")
Pkg.add("StatsPlots")
=#

# Load the installed packages
using CSV
using DataFrames
using GLM
using Plots
using StatsPlots
using Lathe.preprocess: TrainTestSplit
#using MultivariateStats
using StatsModels
using Statistics

struct NeuralNet
  L::Int64                        # number of layers
  n::Vector{Int64}                # sizes of layers
  h::Vector{Vector{Float64}}      # units field
  ξ::Vector{Vector{Float64}}      # units activation
  w::Vector{Array{Float64,2}}     # weights
  θ::Vector{Vector{Float64}}      # thresholds
  delta::Vector{Vector{Float64}}  # propagation errors
  d_w::Vector{Array{Float64,2}}   # changes of weights
  d_θ::Vector{Vector{Float64}}   # changes of thresholds
end

struct Dataset
    names::Vector{Symbol}          # names of the features/ cols
    features::Int64                # number of features
    patterns::Int64                # number of patterns
    boundary::Float64              # percentage of patterns used for training-validation set [0,1]
    train::Array{Float64,2}        # training-validation set
    test::Array{Float64,2}         # test set
    train_df::DataFrame            # training-validation set - Dataframe version
    test_df::DataFrame             # test set - Dataframe version
    rangesTrain::Vector{Tuple}
    rangesTest::Vector{Tuple}
end

# for columns of a DataFrame
function getRanges(df::DataFrame)
  ranges = Vector{Tuple}()
  cols = size(df, 2)

  for i in 1:cols
    tuple = (minimum(df[:,i]), maximum(df[:,i]))
    push!(ranges, tuple)
  end
  return ranges
end

function scale(df::DataFrame, ranges::Vector{Tuple}, s_min::Float64 = 0.0, s_max::Float64 = 1.0)
  # outliers’ detection, treatment of missing data,
  # transformation of categorical data into an appropriate numeric representation, etc
  rows = size(df, 1)
  cols = size(df, 2)

  #ranges = getRanges(df)

  for i in 1:rows
    for j in 1:cols
      x = df[i,j]
      x_min = ranges[j][1]
      x_max = ranges[j][2]
      s = s_min + ((s_max - s_min)/(x_max - x_min)) * (x - x_min)
      df[i,j] = s
    end
  end
end

function descale(df::DataFrame, ranges::Vector{Tuple},s_min::Float64 = 0.0, s_max::Float64 = 1.0)
  # outliers’ detection, treatment of missing data,
  # transformation of categorical data into an appropriate numeric representation, etc
  rows = size(df, 1)
  cols = size(df, 2)

  #ranges = getRanges(df)

  for i in 1:rows
    for j in 1:cols
      s = df[i,j]
      x_min = ranges[j][1]
      x_max = ranges[j][2]
      x = x_min + ((x_max - x_min)/(s_max - s_min)) * (s - s_min)
      df[i,j] = x
    end
  end
end

# slicer : [0,1] with 1 = 100%, default 80 %
function DataSlicer(path::String, boundary::Float64 = 0.8)
  println("...DataSlicer()")
  df = DataFrame(CSV.File(path))
  rows = size(df, 1)
  cols = size(df, 2)
  names = propertynames(df)
  
  train, test = TrainTestSplit(df, boundary)
  rangesTest = getRanges(test)
  rangesTrain = getRanges(train)

  #dataset = Dataset(names, cols, rows, boundary, Matrix(train), Matrix(test), train, test, rangesTrain, rangesTest)
  scale(train, rangesTrain)
  scale(test, rangesTest)
  
  println("Features : ", cols, " | Patterns : ", rows, " | Boundary : ", boundary, " | Training : ", size(train, 1), " | Test : ", size(test, 1))

  return Dataset(names, cols, rows, boundary, Matrix(train), Matrix(test), train, test, rangesTrain, rangesTest)
end

