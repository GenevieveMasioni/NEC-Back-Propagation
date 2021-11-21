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

    path = args[1]
    data_path = ""
    boundary = 0.8
    folds = 4
    layers = Vector{Int64}()
    epochs = 0
    η = 0.15
    α = 0.45

    open(path) do file
      # line_number
      line = 0

      # read till end of file
      while !eof(file)
        # read a new / next line for every iteration
        s = readline(file)
        line += 1
        println("$line . $s")
        if line == 1
          data_path = s
        elseif line == 2
          chunks = split(s, ' ')
          boundary = parse(Float64, chunks[1])
        elseif line == 3
          folds = parse(Int64, s)
        elseif line == 4
          chunks = split(s, ' ')
          layers = [parse(Int,x) for x in chunks]
        elseif line == 5
          epochs = parse(Int64, s)
        elseif line == 6
          chunks = split(s, ' ')
          η = parse(Float64, chunks[1])
          α = parse(Float64, chunks[2])
        end
      end
    end

    preprocess = false
    if !occursin("A1-turbine.txt",data_path) && !occursin("A1-synthetic.txt",data_path)
      preprocess = true
    end

    data = DataSlicer(string(data_path), boundary, preprocess)
    pushfirst!(layers, size(data.train,2)-1)
    nn = NeuralNet(layers)

    error = crossValidation(nn, data, folds, η, α, epochs)
    println("Prediction errors (bp, mlr) : ", error)
end

main(ARGS)
