include("back_propagation.jl")
include("MLR.jl")
include("utils.jl")

println("...STARTING PROGRAM...")
println("Type the file you wish to select:")
filename = readline()

boundary = 0

if filename == "A1-turbine.txt"
    boundary = 0.85
elseif filename == "A1-synthetic.txt"
    boundary = 0.80
else
    boundary = 0.80
end

data = DataSlicer(string("dataset/",filename), boundary)
layers = [size(data.train,2)-1;9;5; 1]
nn = NeuralNet(layers)

η = 0.15
α = 0.45

BP(nn, data, η, α, filename)


#MLR(data)
#error = crossValidation(nn, data, 4)
#println("Prediction errors (bp, mlr) : ", error)
