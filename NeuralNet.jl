include("utils.jl")

function NeuralNet(layers::Vector{Int64})
  L = length(layers)
  n = copy(layers)

  h = Vector{Float64}[]
  ξ = Vector{Float64}[]
  θ = Vector{Float64}[]
  for ℓ in 1:L
    push!(h, zeros(layers[ℓ]))
    push!(ξ, zeros(layers[ℓ]))
    push!(θ, rand(layers[ℓ]))                     # random, but should have also negative values
  end

  w = Array{Float64,2}[]

  push!(w, zeros(1, 1))                           # unused, but needed to ensure w[2] refers to weights between the first two layers

  for ℓ in 2:L
    push!(w, rand(layers[ℓ], layers[ℓ - 1]))     # random, but should have also negative values
  end

  return NeuralNet(L, n, h, ξ, w, θ)
end


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


layers = [4; 9; 5; 1]
nn = NeuralNet(layers)

#=nn.L
nn.n

nn.ξ
nn.ξ[1]
nn.ξ[2]

nn.w
nn.w[2]

x_in  = rand(nn.n[1])
y_out = zeros(nn.n[nn.L])

feed_forward!(nn, x_in, y_out)

y_out=#
