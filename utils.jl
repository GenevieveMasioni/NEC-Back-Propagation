# Import Packages
using Pkg  # Package to install new packages

# Install packages
#=
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("GLM")
Pkg.add("Plots")
Pkg.add("Lathe")
=#

# Load the installed packages
using CSV
using DataFrames
using GLM
using Plots
using Lathe.preprocess: TrainTestSplit

struct Dataset
    features::Int64                # number of features
    patterns::Int64                # number of patterns
    boundary::Float64              # percentage of patterns used for training-validation set [0,1]
    train::Array{Float64,2}            # training-validation set
    test::Array{Float64,2}                # test set
end