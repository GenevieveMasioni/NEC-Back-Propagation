include("back_propagation.jl")
include("MLR.jl")
include("utils.jl")

println("Type the file you wish to select:")
filename = readline()
boundary = 0
#data = DataSlicer("dataset/A1-turbine.txt", 0.85)
if filename == "A1-turbine.txt"
    boundary = 0.85
elseif filename == "A1-synthetic.txt"
    boundary = 0.80
else
    boundary = 0.80
end
data = DataSlicer(string("dataset/",filename), boundary)
layers = [size(data.train,2)-1;5;5; 1]
nn = NeuralNet(layers)

η = 0.02
α = 0.2
BP(nn, data, η, α)

#Multilinear_regression(data)