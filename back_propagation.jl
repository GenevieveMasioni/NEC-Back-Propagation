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
function sigmoidDeriv(ξ::Float64)::Float64
  return ξ*(1-ξ)
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
      nn.delta[nn.L][i] = sigmoidDeriv(nn.ξ[nn.L][i])*(y_out[i] - z)
  end
  for ℓ in nn.L:-1:2
    for j in 1:nn.n[ℓ-1]
      current_error = 0
      for i in 1:nn.n[ℓ]
        current_error += nn.delta[ℓ][i]*nn.w[ℓ][i,j]
      end
      nn.delta[ℓ-1][j] = sigmoidDeriv(nn.ξ[ℓ-1][j]) * current_error
    end
  end
end

#UPDATE THRESHOLDS AND WEIGHTS
function UpdateThresholdWeights(nn::NeuralNet, η::Float64, α::Float64)
  for ℓ in 2:nn.L
    for i in 1:nn.n[ℓ]
      for j in 1:nn.n[ℓ-1]
        nn.d_w[ℓ][i,j] = -η*nn.delta[ℓ][i]*nn.ξ[ℓ-1][j]+ α*nn.d_w[ℓ][i,j]
        nn.w[ℓ][i,j] += nn.d_w[ℓ][i,j]
      end
      nn.d_θ[ℓ][i] = η*nn.delta[ℓ][i] + α*nn.d_θ[ℓ][i]
      nn.θ[ℓ][i] += nn.d_θ[ℓ][i]
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

function BP(nn::NeuralNet, data::Dataset, η::Float64, α::Float64, epoch::Int64)
  println("...Back Propagation()")
  y_out = zeros(nn.n[nn.L])
  y_predTr = zeros(size(data.train, 1))
  y_predTe = zeros(size(data.test, 1))

  y_test= zeros(size(data.test, 1))
  y_train= zeros(size(data.train, 1))
  MSETrain = zeros(epoch)
  MSETest = zeros(epoch)
  #BP Training
  for ℓ in 1:epoch
    for i in 1:size(data.train, 1)


      rndNum = rand(1:size(data.train, 1))
      x_in = data.train[rndNum,1:size(data.train,2)-1]

      z = data.train[rndNum,size(data.train,2)]
      feed_forward!(nn, x_in, y_out)
      BPError(nn, y_out, z)

      UpdateThresholdWeights(nn,η, α)
    end
    #Relative absolute error
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
  end


  #Display the % error over n epochs
  println("Relative absolute error over ", epoch, " Epochs")
  println("Nr. of Epoch: ",epoch)
  println("Relative absolute error Train: ", MSETrain[epoch])
  println("Relative absolute error Test: ",MSETest[epoch])
  # Add column predicted to dataframe, for later scaling
  data_train_df=deepcopy(data.train_df)
  data_test_df=deepcopy(data.test_df)
  insertcols!(data_train_df, size(data.train,2)+1, :predictedY => y_predTr[:])
  insertcols!(data_test_df, size(data.test,2)+1, :predictedY => y_predTe[:])
  #Scaling to original size
  descale(data_train_df, data.rangesTrain)
  descale(data_test_df, data.rangesTest)
  
  #saving the results on a cvs file
  s = size(data_test_df,2)
  CSV.write(string("Results/BP/_results_test.csv"), data_test_df[:,s-1:s])

  #Plotting Original Output vs Predicted Output
  figureRPTr = scatter(data_train_df[:,size(data_train_df,2)-1],data_train_df[:,size(data_train_df,2)],title = "Predicted Vs Original Train", ylabel="Prediction", xlabel="Original")
  display(figureRPTr)
  figureRPTe = scatter(data_test_df[:,s-1],data_test_df[:,s],title = "Predicted Vs Original Test", ylabel="Prediction", xlabel="Original")
  display(figureRPTe)
  #Plots Of the %Errors
  figureMSETR = plot(MSETrain, title = "Training % Error over Epochs", xlabel="Epoch", ylabel="%Error")
  figureMSETE = plot(MSETest, title = "Test % Error over Epochs", xlabel="Epoch", ylabel="%Error")
  display(figureMSETR)
  display(figureMSETE)

  #save Plots
  png(figureRPTr,string("Plots/BP/figure_Real_Predict_Train.jpg"))
  png(figureRPTe,string("Plots/BP/figure_Real_Predict_Test.jpg"))
  png(figureMSETR,string("Plots/BP/figure_Error_Train.jpg"))
  png(figureMSETE,string("Plots/BP/figure_Error_Test.jpg"))
  #readline()
  return Base.sum(MSETest) / epoch 
end
