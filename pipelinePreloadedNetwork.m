%% Load Network
load NetworkEquationRecognition.mat

%% Test Network

userInput = input('Please enter the complete path to the equation images folder. ','s');
[equationNames, equationLabels] = loadEquationImagesFromFolder(userInput);

display('Loading and testing...')

% Pass each equation file name to segment and propogate images
% Returns location of segment and class
for i = 1:length(equationNames)
    [Character, class, mergeClass] = propagateEquationsThroughNetwork(equationNames{i},...
        hiddenWeightsLetters, hiddenWeightsNumbers, hiddenWeightsSymbols,...
        outputWeightsLetters, outputWeightsNumbers,outputWeightsSymbols,...
        tempLowerLetters, tempDigits, tempSymbols);
    
    % Writes to file
    writeEquationsToFile(equationLabels{i},Character,class)
end