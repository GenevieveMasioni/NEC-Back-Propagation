include("utils.jl")

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
function BPError(nn::NeuralNet, y_out::Vector{Float64}, z::Float64)#data::Vector{Float64})

  for i in 1:nn.n[nn.L]
    nn.delta[nn.L][i] = sigmoid(nn.h[nn.L][i])*(y_out[i] - z)
  end
  for ℓ in 2:nn.L
    for j in 1:nn.n[ℓ-1]
      current_error = 0
      for i in 1:nn.n[ℓ]
        current_error += nn.delta[ℓ][i]*nn.w[ℓ][i,j]   
      end
      nn.delta[ℓ-1][j] = sigmoid(nn.h[ℓ-1][j]) * current_error
    end
  end
end

#UPDATE THRESHOLDS AND WEIGHTS
function UpdateThresholdWeights(nn::NeuralNet, η::Float64, α::Float64, p::Int64)
  for ℓ in 2:p:nn.L
    for i in 1:nn.n[ℓ]
      for j in 1:nn.n[ℓ-1]
        sumw = 0
        for k in 1:nn.L-1
          if (size(nn.delta[ℓ+k-1],1)>=i)
            if (size(nn.ξ[(ℓ-1)+k-1],1)>=j)
              sumw += nn.delta[ℓ+k-1][i]*nn.ξ[(ℓ-1)+k-1][j]
            end
          else
            break
          end  
        end
        nn.d_w[ℓ][i,j] = -η*sumw + α*nn.d_w[ℓ][i,j]
        nn.w[ℓ][i,j] = nn.w[ℓ][i,j]+nn.d_w[ℓ][i,j]
      end
      sumθ = 0
      for k in 1:nn.L-1
        if (size(nn.delta[ℓ+k-1],1)>=i)
          sumθ += nn.delta[ℓ+k-1][i]
        else
          break
        end
      end
      nn.d_θ[ℓ][i] = η*sumθ + α*nn.d_θ[ℓ][i]
      nn.θ[ℓ][i] = nn.θ[ℓ][i]+nn.d_θ[ℓ][i]
    end
  end
end
#BACK PROPAGATE ALGORITHM

function QuadraticError(y_pred::Vector{Float64}, y_true::Vector{Float64}, nrObservation)
  MSE = 0
  sumY = 0
  for i in size(y_pred,1)
    MSE += abs(y_pred[i]-y_true[i])
    sumY += y_pred[i]
  end

  MSE = (MSE/sumY)*100
  return MSE
end

function BP(nn::NeuralNet, data::Dataset, η::Float64, α::Float64) 
  epoch = 10
  y_out = zeros(nn.n[nn.L])
  y_predTr = zeros(size(data.train, 1))
  y_predTe = zeros(size(data.test, 1))
  batch = 5
  y_test= zeros(size(data.test, 1))
  y_train= zeros(size(data.train, 1))
  MSETrain = zeros(epoch)
  MSETest = zeros(epoch)
 for ℓ in 1:epoch
    for i in 1:batch:size(data.train, 1)
      for j in 1:batch
        if i+batch>size(data.train, 1)
          rndNum = rand(i:size(data.train, 1))
          x_in = data.train[rndNum,1:size(data.train,2)-1]
        else
          rndNum = rand(i:batch+i-1)
          x_in = data.train[rndNum,1:size(data.train,2)-1]
        end
      #rndNum = rand(1:size(data.train, 1))
      #x_in=data.train[rndNum,1:size(data.train,2)-1]
        z = data.train[rndNum,size(data.train,2)]
        feed_forward!(nn, x_in, y_out)  
        BPError(nn, y_out, z)  
      end
      UpdateThresholdWeights(nn,η, α, batch)
    end
    #Quadratic Errors
    println("Quadratic ERROR")
    for k in 1:size(data.train,1)
      x_in = data.train[k,1:size(data.train, 2)-1]
      feed_forward!(nn, x_in, y_out)
      y_predTr[k] = y_out[1]
      
    end
    for k in 1:size(data.test,1)
      x_in = data.test[k,1:size(data.test, 2)-1]
      feed_forward!(nn, x_in, y_out)
      y_predTe[k] = y_out[1]
    end
    y_train = data.train[:,size(data.train, 2)]
    MSETrain[ℓ]= QuadraticError(y_predTr, y_train, size(data.train, 1))

    y_test = data.test[:,size(data.test, 2)]
    MSETest[ℓ]= QuadraticError(y_predTe, y_test, size(data.test, 1))
     
    println("Epoch: ",ℓ," MSETrain: ", MSETrain[ℓ], " MSETest: ",MSETest[ℓ])  
  end
  #Plot real vs predicted
  
  figureRPTr = scatter(y_predTr, y_train)
  figureRPTe = scatter(y_predTe, y_test)
  display(figureRPTr)
  readline()
  display(figureRPTe)
  readline()
  #Plots
  figureMSETR = scatter(MSETrain)
  figureMSETE = scatter(MSETest)
  display(figureMSETR)
  readline()
  display(figureMSETE)
  readline()
  gui()
end
