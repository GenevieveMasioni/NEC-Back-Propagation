include("genevieve.jl")

function NeuralNet(layers::Vector{Int64})
  L = length(layers)
  n = copy(layers)

  h = Vector{Float64}[]
  ξ = Vector{Float64}[]
  θ = Vector{Float64}[]
  delta = Vector{Float64}[]
  d_θ = Vector{Float64}[]
  for ℓ in 1:L
    push!(h, zeros(layers[ℓ]))
    push!(ξ, zeros(layers[ℓ]))
    push!(θ, rand(layers[ℓ]))                     # random, but should have also negative values
    push!(delta, zeros(layers[ℓ]))
    push!(d_θ, zeros(layers[ℓ]))
  end

  w = Array{Float64,2}[]
  d_w = Array{Float64,2}[]
  

  push!(w, zeros(1, 1))                           # unused, but needed to ensure w[2] refers to weights between the first two layers
  push!(d_w, zeros(1,1))
  

  for ℓ in 2:L
    push!(w, rand(layers[ℓ], layers[ℓ - 1]))     # random, but should have also negative values
    push!(d_w, zeros(layers[ℓ],layers[ℓ - 1]))
  end

  return NeuralNet(L, n, h, ξ, w, θ, delta, d_w, d_θ)
end

# FEED FOWARD PROPAGATION
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
function BPError(nn::NeuralNet, y_out::Vector{Float64}, data::Vector{Float64})
  current_error = 0

  for i in 1:nn.n[nn.L]
    nn.delta[nn.L][i] = sigmoid(nn.h[nn.L][i])*(y_out[i]- data[i])
  end
  for ℓ in 2:nn.L
    for j in 1:nn.n[ℓ-1]
      for i in 1:nn.n[ℓ]
        current_error += nn.delta[ℓ][i]*nn.w[ℓ][i,j]
      end
      nn.delta[ℓ-1][j] = sigmoid(nn.h[ℓ-1][j]) * current_error
    end
  end

end

#UPDATE THRESHOLDS AND WEIGHTS
function UpdateThresholdWeights(nn::NeuralNet, η::Float64, α::Float64)

  for ℓ in 2:nn.L
    for i in 1:nn.n[ℓ]
      for j in 1:nn.n[ℓ-1]
        nn.d_w[ℓ][i,j] = -η*nn.delta[ℓ][i]*nn.ξ[ℓ-1][j] + α*nn.d_w[ℓ][i,j]
        nn.w[ℓ][i,j] = nn.w[ℓ][i,j]+nn.d_w[ℓ][i,j]
      end
      nn.d_θ[ℓ][i] = η*nn.delta[ℓ][i] + α*nn.d_θ[ℓ][i]
      nn.θ[ℓ][i] = nn.θ[ℓ][i]+nn.d_θ[ℓ][i]
    end
  end

end
#BACK PROPAGATE ALGORITHM

function BP(nn::NeuralNet, data::Dataset, η::Float64, α::Float64) #this section need to be looked at (see how to make use of the dataframes)
  epoch = 10
  y_out = zeros(nn.n[nn.L])
 for ℓ in 1:epoch
    for j in 1:size(data.train, 1)
      #random pattern
      pattern = data.train[rand(1:size(data.train, 1)),:]
      feed_forward!(nn, pattern, y_out) #data should be a vector float
      BPError(nn, y_out, pattern)
      UpdateThresholdWeights(nn,η, α)
    end

  end
end

layers = [5; 9; 5; 1]
nn = NeuralNet(layers)
data = DataSlicer("dataset/A1-turbine.txt", 0.85)

η = 0.01
α = 0.1
#print(data.train[rand(1:size(data.train, 1)),1:size(data.train, 2)-1])
BP(nn, data, η, α)
