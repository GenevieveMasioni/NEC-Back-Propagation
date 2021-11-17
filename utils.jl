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
