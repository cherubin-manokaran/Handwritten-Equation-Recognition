% Derivative of Logistic Sigmoid Function
function values = dActivationFunction(input)
    values = input.*(1-input);
end