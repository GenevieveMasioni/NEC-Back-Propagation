using CSV
using DataFrames
# CALL OF FILES
#df = CSV.read("A1-synthetic.txt", DataFrame)
#df = DataFrame(CSV.File("A1-synthetic.txt"))
#print(df)

#NEURONAL NETWORK ARCHITECTURE
struct NeuralNet
    L::Int64                        # number of layers
    n::Vector{Int64}                # sizes of layers
    h::Vector{Vector{Float64}}      # units field
    ξ::Vector{Vector{Float64}}      # units activation
    w::Vector{Array{Float64,2}}     # weights
    θ::Vector{Vector{Float64}}      # thresholds
    delta::Vector{Vector{Float64}}  # propagation errors
    d_w::Vector{Array{Float64,2}}   # changes of weights
    d_θ::Vector{Array{Float64,2}}   # changes of thresholds
    
end


function NeuralNet(layers::Vector{Int64})
    L = length(layers)
    n = copy(layers)
  
    h = Vector{Float64}[]
    ξ = Vector{Float64}[]
    θ = Vector{Float64}[]
    delta = Vector{Float64}[]
    for ℓ in 1:L
      push!(h, zeros(layers[ℓ]))
      push!(ξ, zeros(layers[ℓ]))
      push!(θ, rand(layers[ℓ]))                     # random, but should have also negative values
      push!(delta, zeros(layers[ℓ]))
      push!(d_w, zeros(layers[ℓ]))
      push!(d_θ, zeros(layers[ℓ]))                   
    end
  
    w = Array{Float64,2}[]
    d_w = Array{Float64,2}[]
    push!(w, zeros(1, 1))                           # unused, but needed to ensure w[2] refers to weights between the first two layers
                              
    for ℓ in 2:L
      push!(w, rand(layers[ℓ], layers[ℓ - 1]))     # random, but should have also negative values
    end
  
    return NeuralNet(L, n, h, ξ, w, θ)
end

#FEED FOWARD PROPAGATION
function sigmoid(h::Float64)::Float64
    return 1 / (1 + exp(-h))
  end
  
  
function feed_forward!(nn::NeuralNet, x_in::Vector{Float64}, y_out::Vector{Float64})
    # copy input to first layer, Eq. (6)
    nn.ξ[1] .= x_in
  
    # feed-forward of input pattern
    for ℓ in 2:nn.L
      for i in 1:nn.n[ℓ]
        # calculate input field to unit i in layer ℓ, Eq. (8)
        h = -nn.θ[ℓ][i]
        for j in 1:nn.n[ℓ - 1]
          h += nn.w[ℓ][i, j] * nn.ξ[ℓ - 1][j]
        end
        # save field and calculate activation, Eq. (7)
        nn.h[ℓ][i] = h
        nn.ξ[ℓ][i] = sigmoid(h)
      end
    end
  
    # copy activation in output layer as output, Eq. (9)
    y_out .= nn.ξ[nn.L]
end

#BACK PROPAGATE ERROR
function BPError(nn::NeuralNet, y_out::Vector{Float64}, dfTemp, count::Int64)

    for ℓ in 2:nn.L
        
        if ℓ == nn.L 
            nn.delta[L] .= (nn.h[L]) * (y_out-dfTemp)
        else
            for i in 1:nn.n[ℓ]
                nn.delta[ℓ-1].=nn.delta[ℓ]*nn.w[ℓ][i]
            end
        end
    end

end

#UPDATE THRESHOLDS AND WEIGHTS
function UpdateThresholdWeights(nn::NeuralNet,count::Int64)
    nn.d_w[count].=-nn.delta[count]*nn.ξ[count-1]+nn.d_w[count-1]
    nn.d_θ[count].=nn.delta[count] + nn.d_θ[count-1]

    nn.w[count] = nn.w[count-1]+nn.d_w[count]
    nn.θ[count] = nn.θ[count-1]+nn.d_θ[count] 
end
#BACK PROPAGATE ALGORITHM

function BP(nn::NeuralNet, dfTemp)
  for ℓ in 2:nn.L
    for i in 1:nn.n[ℓ]
      #choose rnd pattern of the training set

      #feed_forward
      feed_forward()
      BPError()
      UpdateThresholdWeights()

    end
      
  end
end

