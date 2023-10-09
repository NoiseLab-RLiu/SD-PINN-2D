function U = model_2D(parameters,X,Y,T)
XYT = [X;Y;T];
numLayers = numel(fieldnames(parameters));
% First fully connect operation.
weights = parameters.fc1.Weights;
bias = parameters.fc1.Bias;
U = fullyconnect(XYT,weights,bias);

% tanh and fully connect operations for remaining layers.
for i=2:5
    name = "fc" + i;

    U = tanh(U);

    weights = parameters.(name).Weights;
    bias = parameters.(name).Bias;
    U = fullyconnect(U, weights, bias);
end
end