# Back Propagation

Neural and Evolutionary Computation (NEC) project : prediction with Back-Propagation (BP) and Multiple Linear Regression (MLR).

## Objective

Prediction using the following algorithms:
• Back-Propagation (BP), implemented by the student
• Multiple Linear Regression (MLR), using free software

## Tasks list
Functions :
- [ ] Data preprocessing (third dataset)
- [ ] Data normalization (all datasets)
- [ ] Cross-validation (MLR and BP)
- [ ] BP
- [ ] MLR (use of an existing Julia lib)
- [ ] Compute performance (automatisation of test process to find the best parameters for relative absolute error minimization)
- [x] Data slicer (training-validation and test)

## Datasets

The predictions must be performed on three datasets:
1. File: A1-turbine.txt
- 5 features: the first 4 are the input variables, the last one is the value to predict
- 451 patterns: use the first 85% for training and validation, and the remaining 15% for test
2. File: A1-synthetic.txt
- 10 features: the first 9 are the input variables, the last one is the value to predict
- 1000 patterns: use the first 80% for training and validation, and the remaining 20% for test
3. Dataset from the Internet:
- 6 features, one of them used for prediction (not a categorical value)
- At least 400 patterns
- 80% of the patterns for training and validation, 20% for test. Shuffled data to destroy any kind of sorting.

## Procedure

- Data preprocessing of the third dataset.
- Apply cross-validation (n-fold cross-validation, leave-1-out, bootstrapping, etc.) for both  MLR  and  BP, report  the  expected  prediction  error obtained from cross-validation, and compare it with the prediction error on the test set.
- Find good values for all the parameters of BP: architecture of the network, learning  rate  and  momentum,  activation function,  and  number  of  epochs.
- Automate the whole process.

## Implementation of BP

- Language : Julia
- The  code  is able  to  deal  with  arbitrary multilayer  networks. For  example,  a  network  with  architecture  3:9:5:1  (4  layers,  3
input  units,  1  output  unit,  and  two  hidden  layers  with  9  and  5  units,  respectively).

## Multilinear regression

- Ready to use Julia library.
- Cross-validation  to estimate its prediction capability.

## Evaluation of the results

Rather than minimization of the mean squared error, use of the relative absolute error (or percentage of the error) to evaluate the performance on the test set.

## Training parameters and execution

All the training parameters must be put in a text file. It must include:
- Name of the data file
- Number of training and test patterns
- Information about the cross-validation (number of folds, and/or percentage of training patterns used for  
- Number of layers
- Number of units in each layer
- Number of epochs
- Learning rate and momentum
- Optionally: information about the scaling method (normalization or standardization) of inputs and/or outputs, and in the case of normalization, the range of the normalized data
- Optionally: the selected activation function (sigmoid, tanh, ReLU, etc.)
- Optionally: name of output file(s)
