include("utils.jl")
include("back_propagation.jl")
include("MLR.jl")

function main(args)
    @show args
    if(size(args,1) < 1)
      println("Error : the program expects 1 argument.")
      println("Usage : julia run_bp.jl parameters_file_path")
      return;
    elseif (size(args,1) > 1)
      println("Warning : too many arguments (1 expected). Only 1 will be considered.")
    end

    filename = args[1]

    # TODO : load args from parameters_file.txt
    boundary = 0
    if occursin("A1-turbine.txt",filename)
        boundary = 0.85
    elseif occursin("A1-synthetic.txt",filename)
        boundary = 0.80
    else
        boundary = 0.80
    end

    data = DataSlicer(string(filename), boundary)
    layers = [size(data.train,2)-1;9;5; 1]
    nn = NeuralNet(layers)

    
    #=η = 0.15
    α = 0.45
    BP(nn, data, η, α, filename)=#
    

    #MLR(data)
    #error = crossValidation(nn, data, 4)
    #println("Prediction errors (bp, mlr) : ", error)
end

main(ARGS)
