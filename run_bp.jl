include("back_propagation.jl")
include("MLR.jl")
include("utils.jl")

data = DataSlicer("dataset/A1-turbine.txt", 0.85)
layers = [size(data.train,2)-1; 9; 5; 1]
nn = NeuralNet(layers)

η = 0.05
α = 0.1

BP(nn, data, η, α)

#Multilinear_regression(data)