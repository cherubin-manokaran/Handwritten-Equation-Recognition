function [hiddenWeightsLetters, hiddenWeightsNumbers, hiddenWeightsSymbols,...
    outputWeightsLetters, outputWeightsNumbers, outputWeightsSymbols,...
    tempLowerLetters, tempDigits, tempSymbols] = trainNetwork()   
    %% Isolate Letter, Number, and Symbol Images
    %folder = 'C:\Users\Manokaran\Documents\MATLAB\annotated';

    folder = input('Please enter the complete path to the training images folder. ','s');
    
    if ~isdir(folder)
      errorMessage = sprintf('Error: The following folder does not exist:\n%s', folder);
      uiwait(warndlg(errorMessage));
      return;
    end
    
    display('Loading images...')

    if ~isdir(folder)
      errorMessage = sprintf('Error: The following folder does not exist:\n%s', folder);
      uiwait(warndlg(errorMessage));
      return;
    end

    filePattern = fullfile(folder, '*.png');
    pngFiles = dir(filePattern);
    files = {pngFiles.name};
    names = cell(length(files),1);
    images = (zeros(28,28,length(files)));

    lowerAlphabet = 'a':'z';
    upperAlphabet = 'A':'Z';
    numbers = 0:9;
    symbols = {'(',')','=','+','-','div','frac','mul','delta','pm','sqrt','bar','dots','pi','tan','cos','sin'};

    targetLowerLetterValues = zeros(length(lowerAlphabet), length(files));
    targetDigitValues = zeros(length(numbers),length(files));
    targetSymbolValues = zeros(length(symbols),length(files));
    targetUpperLetterValues = zeros(length(upperAlphabet), length(files));

    % Stores all names of files
    % Accesses images in natural order
    % Resizes images to MNIST data dimensions

    for k = 1:length(files)
        fullFileName = fullfile(folder, files{k});
        [~,name,ext] = fileparts(fullFileName);
        names{k} = name;
        filename = strcat(name, ext);

        for l = 1:length(lowerAlphabet)
            patternTrue = strfind(name, sprintf('_%s_', lowerAlphabet(l)));
            if (length(patternTrue) >= 1)
              targetLowerLetterValues(l, k) = 1;
            end
        end

        for l = 1:length(numbers)
            patternTrue = strfind(name, sprintf('_%d_', numbers(l)));
            if (length(patternTrue) >= 1)
              targetDigitValues(l, k) = 1;
            end
        end

        for l = 1:length(symbols)
            patternTrue = strfind(name, sprintf('_%s_', symbols{l}));
            if (length(patternTrue) >= 1)
              targetSymbolValues(l, k) = 1;
            end
        end

        for l = 1:length(upperAlphabet)
            patternTrue = strfind(name, sprintf('_%s_', upperAlphabet(l)));
            if (length(patternTrue) >= 1)
                targetUpperLetterValues(l, k) = 1;
            end
        end

        fullFileName = fullfile(folder, filename);
        image = imread(fullFileName);

        filter = rand(4);
        output_temp = conv2(double(image),double(filter));
        output = bilinearInterpolation(output_temp, [28 28]);

        images(:,:,k) = output;
        out = zeros(28);
    end

    figure
    imshow(image)

    figure
    imshow(output_temp)

    figure
    imshow(output)

    images = reshape(images, size(images, 1) * size(images, 2), size(images,3));
    images = double(images) / 255;

    [rows, columns] = find(targetLowerLetterValues);
    tempLowerLetters = targetLowerLetterValues(:, columns);
    tempLowerLetterImages = images(:, columns);

    [rows, columns] = find(targetDigitValues);
    tempDigits = targetDigitValues(:, columns);
    tempDigitImages = images(:, columns);

    [rows, columns] = find(targetSymbolValues);
    tempSymbols = targetSymbolValues(:, columns);
    tempSymbolImages = images(:, columns);

    [rows, columns] = find(targetUpperLetterValues);
    tempUpperLetters = targetUpperLetterValues(:, columns);
    tempUpperLetterImages = images(:, columns);

    display('Training network...')

    %% Train letters using the isolated set

    trainingSet = length(tempLowerLetters(1,:));
    inputDimensions = length(images(:,1));
    outputSize = length(tempLowerLetters(:,1));

    hiddenUnits = 500;
    learningRate = 0.2;
    momentum = 0.1;

    hiddenWeightsLetters = rand(hiddenUnits, inputDimensions);
    hiddenWeightsLetters = hiddenWeightsLetters/length(hiddenWeightsLetters(1,:));
    outputWeightsLetters = rand(outputSize, hiddenUnits);
    outputWeightsLetters = outputWeightsLetters/length(outputWeightsLetters(1,:));

    epochs = 1:100;
    batch = 15;

    output = zeros(10,1);
    errorVec = zeros(size(epochs));

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    index = 1;

    for i = 1:length(epochs)
        error = 0;
        for j = 1:batch
            n(batch*(i-1)+j) = floor(rand(1) * trainingSet + 1);
            inputVector = tempLowerLetterImages(:, n(batch*(i-1)+j));
            targetVector = tempLowerLetters(:, n(batch*(i-1)+j));

            [H1, output] = propogateInput(hiddenWeightsLetters, outputWeightsLetters, inputVector);
            alloutput(:, n(batch*(i-1)+j)) = output;

            [hiddenWeightsLetters, outputWeightsLetters, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                momentum, hiddenWeightsLetters, outputWeightsLetters, inputVector, targetVector, H1, ...
                output, prevdOutWeight, prevdHiddenWeight);

            prevdOutWeight = dOutWeight;
            prevdHiddenWeight = dHiddenWeight;

            error = error + norm(output - targetVector,2);

            index = index + 1;
        end
        errorVec(i) = error/batch;
    end

    %% Test network with all images backpropogating if necessary
    % Particularly when letter image and when output is not clear 

    testSet = length(images(1,:));
    outVec3 = zeros(length(tempLowerLetters(:,1)), testSet);

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeightsLetters, outputWeightsLetters, inputVector);
        outVec3(:,i) = output;

        update = (max(targetLowerLetterValues(:,i)) == 1);

        if (max(output) < 0.97 && update == 1)
            targetVector = targetLowerLetterValues(:, i);
            [hiddenWeightsLetters, outputWeightsLetters, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                        momentum, hiddenWeightsLetters, outputWeightsLetters, inputVector, targetVector, H1, ...
                        output, prevdOutWeight, prevdHiddenWeight);

            prevdOutWeight = dOutWeight;
            prevdHiddenWeight = dHiddenWeight;

            error = error + norm(output - targetVector,2);

            index = index + 1;
        end
    end

    %% Test network again, mostly for troubleshooting purposes

    testSet = length(images(1,:));
    outVec = zeros(length(tempLowerLetters(:,1)), testSet);

    correctlyClassified = 0;
    incorrectlyClassified = 0;

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeightsLetters, outputWeightsLetters, inputVector);
        outVec(:,i) = output;

        update = (max(targetLowerLetterValues(:,i)) == 1);

        [~, indexOfMax1] = max(targetLowerLetterValues(:,i));
        [~, indexOfMax2] = max(output);

        if (update == 1)
            if (indexOfMax1 == indexOfMax2)
                correctlyClassified = correctlyClassified + 1;
            else
                incorrectlyClassified = incorrectlyClassified + 1;
            end
        end
    end

    display(['Correctly Classified Letters in Testing: ', num2str(correctlyClassified)])
    display(['Incorrectly Classified Letters in Testing: ', num2str(incorrectlyClassified)])

    %% Train numbers using isolated set

    trainingSet = length(tempDigits(1,:));
    inputDimensions = length(images(:,1));
    outputSize = length(tempDigits(:,1));

    hiddenUnits = 500;
    learningRate = 0.2;
    momentum = 0.2;

    hiddenWeightsNumbers = rand(hiddenUnits, inputDimensions);
    hiddenWeightsNumbers = hiddenWeightsNumbers/length(hiddenWeightsNumbers(1,:));
    outputWeightsNumbers = rand(outputSize, hiddenUnits);
    outputWeightsNumbers = outputWeightsNumbers/length(outputWeightsNumbers(1,:));

    epochs = 1:40;
    batch = 10;

    output = zeros(10,1);
    errorVec = zeros(size(epochs));

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    index = 1;
    for i = 1:length(epochs)
        error = 0;
        for j = 1:batch
            n(batch*(i-1)+j) = floor(rand(1) * trainingSet + 1);
            inputVector = tempDigitImages(:, n(batch*(i-1)+j));
            targetVector = tempDigits(:, n(batch*(i-1)+j));

            [H1, output] = propogateInput(hiddenWeightsNumbers, outputWeightsNumbers, inputVector);
            alloutput2(:, n(batch*(i-1)+j)) = output;

            [hiddenWeightsNumbers, outputWeightsNumbers, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                momentum, hiddenWeightsNumbers, outputWeightsNumbers, inputVector, targetVector, H1, ...
                output, prevdOutWeight, prevdHiddenWeight);

            prevdOutWeight = dOutWeight;
            prevdHiddenWeight = dHiddenWeight;

            error = error + norm(output - targetVector,2);

            index = index + 1;
        end
        errorVec(i) = error/batch;
    end

    %% Test network with all images backpropogating if necessary
    % Particularly when number image and when output is not clear 

    testSet = length(images(1,:));
    outVec2 = zeros(length(tempDigits(:,1)), testSet);

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeightsNumbers, outputWeightsNumbers, inputVector);
        outVec2(:,i) = output;

        update = (max(targetDigitValues(:,i)) == 1);

        if (max(output) < 0.97 && update == 1)
            targetVector = targetDigitValues(:, i);
            [hiddenWeightsNumbers, outputWeightsNumbers, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                        momentum, hiddenWeightsNumbers, outputWeightsNumbers, inputVector, targetVector, H1, ...
                        output, prevdOutWeight, prevdHiddenWeight);

            prevdOutWeight = dOutWeight;
            prevdHiddenWeight = dHiddenWeight;

            error = error + norm(output - targetVector,2);

            index = index + 1;
        end
    end

    %% Test nework again, mostly for troubleshooting purposes

    testSet = length(images(1,:));
    outVec2 = zeros(length(tempDigits(:,1)), testSet);

    correctlyClassified = 0;
    incorrectlyClassified = 0;

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeightsNumbers, outputWeightsNumbers, inputVector);
        outVec2(:,i) = output;

        update = (max(targetDigitValues(:,i)) == 1);

        [~, indexOfMax1] = max(targetDigitValues(:,i));
        [~, indexOfMax2] = max(output);

        if (update == 1)
            if (indexOfMax1 == indexOfMax2)
                correctlyClassified = correctlyClassified + 1;
            else
                incorrectlyClassified = incorrectlyClassified + 1;
            end
        end
    end

    display(['Correctly Classified Numbers in Testing: ', num2str(correctlyClassified)])
    display(['Incorrectly Classified Numbers in Testing: ', num2str(incorrectlyClassified)])

    %% Train symbols using isolated set

    trainingSet = length(tempSymbols(1,:));
    inputDimensions = length(images(:,1));
    outputSize = length(tempSymbols(:,1));

    hiddenUnits = 500;
    learningRate = 0.2;
    momentum = 0.2;

    hiddenWeightsSymbols = rand(hiddenUnits, inputDimensions);
    hiddenWeightsSymbols = hiddenWeightsSymbols/length(hiddenWeightsSymbols(1,:));
    outputWeightsSymbols = rand(outputSize, hiddenUnits);
    outputWeightsSymbols = outputWeightsSymbols/length(outputWeightsSymbols(1,:));

    epochs = 1:70;
    batch = 20;

    output = zeros(10,1);
    errorVec = zeros(size(epochs));

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    index = 1;
    for i = 1:length(epochs)
        error = 0;
        for j = 1:batch
            n(batch*(i-1)+j) = floor(rand(1) * trainingSet + 1);
            inputVector = tempSymbolImages(:, n(batch*(i-1)+j));
            targetVector = tempSymbols(:, n(batch*(i-1)+j));

            [H1, output] = propogateInput(hiddenWeightsSymbols, outputWeightsSymbols, inputVector);
            alloutput3(:, n(batch*(i-1)+j)) = output;

            [hiddenWeightsSymbols, outputWeightsSymbols, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                momentum, hiddenWeightsSymbols, outputWeightsSymbols, inputVector, targetVector, H1, ...
                output, prevdOutWeight, prevdHiddenWeight);

            prevdOutWeight = dOutWeight;
            prevdHiddenWeight = dHiddenWeight;

            error = error + norm(output - targetVector,2);

            index = index + 1;
        end
        errorVec(i) = error/batch;
    end

    %% Test network with all images backpropogating if necessary
    % Particularly when symbol image and when output is not clear 

    testSet = length(images(1,:));
    outVec3 = zeros(length(tempSymbols(:,1)), testSet);

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeightsSymbols, outputWeightsSymbols, inputVector);
        outVec3(:,i) = output;

        update = (max(targetSymbolValues(:,i)) == 1);

        if (max(output) < 0.97 && update == 1)
            targetVector = targetSymbolValues(:, i);
            [hiddenWeightsSymbols, outputWeightsSymbols, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                        momentum, hiddenWeightsSymbols, outputWeightsSymbols, inputVector, targetVector, H1, ...
                        output, prevdOutWeight, prevdHiddenWeight);

            prevdOutWeight = dOutWeight;
            prevdHiddenWeight = dHiddenWeight;

            error = error + norm(output - targetVector,2);

            index = index + 1;
        end
    end

    %% Test network again, mostly for troubleshooting purposes

    testSet = length(images(1,:));
    outVec3 = zeros(length(tempSymbols(:,1)), testSet);

    correctlyClassified = 0;
    incorrectlyClassified = 0;

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeightsSymbols, outputWeightsSymbols, inputVector);
        outVec3(:,i) = output;

        update = (max(targetSymbolValues(:,i)) == 1);

        [~, indexOfMax1] = max(targetSymbolValues(:,i));
        [~, indexOfMax2] = max(output);

        if (update == 1)
            if (indexOfMax1 == indexOfMax2)
                correctlyClassified = correctlyClassified + 1;
            else
                incorrectlyClassified = incorrectlyClassified + 1;
            end
        end
    end

    display(['Correctly Classified Symbols in Testing: ', num2str(correctlyClassified)])
    display(['Incorrectly Classified Symbols in Testing: ', num2str(incorrectlyClassified)])

    %%

    trainingSet = length(tempUpperLetters(1,:));
    inputDimensions = length(images(:,1));
    outputSize = length(tempUpperLetters(:,1));

    hiddenUnits = 500;
    learningRate = 0.2;
    momentum = 0.1;

    hiddenWeights4 = rand(hiddenUnits, inputDimensions);
    hiddenWeights4 = hiddenWeights4/length(hiddenWeights4(1,:));
    outputWeights4 = rand(outputSize, hiddenUnits);
    outputWeights4 = outputWeights4/length(outputWeights4(1,:));

    epochs = 1:103;
    batch = 15;

    output = zeros(10,1);
    errorVec = zeros(size(epochs));

    prevdOutWeight = 0;
    prevdHiddenWeight = 0;

    index = 1;
    %for h = 1:2
        for i = 1:length(epochs)
            error = 0;
            for j = 1:batch
                n(batch*(i-1)+j) = floor(rand(1) * trainingSet + 1);
                inputVector = tempUpperLetterImages(:, n(batch*(i-1)+j));
                targetVector = tempUpperLetters(:, n(batch*(i-1)+j));

                [H1, output] = propogateInput(hiddenWeights4, outputWeights4, inputVector);
                alloutput4(:, n(batch*(i-1)+j)) = output;

                [hiddenWeights4, outputWeights4, dOutWeight, dHiddenWeight] = backPropogateError(learningRate,...
                    momentum, hiddenWeights4, outputWeights4, inputVector, targetVector, H1, ...
                    output, prevdOutWeight, prevdHiddenWeight);

                prevdOutWeight = dOutWeight;
                prevdHiddenWeight = dHiddenWeight;

                error = error + norm(output - targetVector,2);

                index = index + 1;
            end
            errorVec(i) = error/batch;
        end
    %end

    %%
    testSet = length(images(1,:));
    outVec4 = zeros(length(tempUpperLetters(:,1)), testSet);

    for i = 1:testSet
        inputVector = images(:,i);
        [H1, output] = propogateInput(hiddenWeights4, outputWeights4, inputVector);
        outVec4(:,i) = output;
    end
end