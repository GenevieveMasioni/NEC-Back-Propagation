# Import Packages
using Pkg  # Package to install new packages

# Install packages
#=
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("GLM")
Pkg.add("Plots")
Pkg.add("Lathe")
Pkg.add("MultivariateStats")
=#

# Load the installed packages
using CSV
using DataFrames
using GLM
using Plots
using Lathe.preprocess: TrainTestSplit
using MultivariateStats
