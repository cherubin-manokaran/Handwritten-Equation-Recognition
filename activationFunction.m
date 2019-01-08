% Logistic Sigmoid Function
% Calculate output layer given input
function values = activationFunction(input)
    values = 1./(1+exp(-input));
end