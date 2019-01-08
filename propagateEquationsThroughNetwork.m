function [Character, class, mergeClass] = propagateEquationsThroughNetwork(filename, ...
    hiddenWeightsLetters, hiddenWeightsNumbers, hiddenWeightsSymbols,...
    outputWeightsLetters, outputWeightsNumbers, outputWeightsSymbols,...
    tempLowerLetters, tempDigits, tempSymbols)
    %% Segment Images
    %filename = 'C:\Users\Manokaran\Documents\MATLAB\annotated\SKMBT_36317040717363_eq13.png';
    [Character, indices] = segmenter(filename);

    numBoxes = size(Character,2);
    Im = imread(filename);
    
%     filter = rand(4);
%     Im = conv2(double(Im_temp),double(filter));

    a = 28;
    b = 28;

    %output matrix O
    fullImages = zeros(a,b,numBoxes);
    images = zeros(a*b,numBoxes);

    for l = 1:numBoxes
        someBox = Character(l).BoundingBox();
        box = floor(someBox); 

        %ensure bounding box is within the image 
        b1 = box(1); 
        b2 = box(1)+box(3); 
        b3 = box(2);
        b4 = box(2)+box(4);
        if(b1 < 1)
            b1 = 1;
        end
        if(b2 > size(Im,2))
            b2 = size(Im,1);
        end
        if(b3 < 1)
            b3 = 1;
        end
        if(b4 > size(Im,1))
            b4 = size(Im,2);
        end
        if(box(3) > 400 || box(4) > 400)
            continue
        end

        I_temp = Im(b3:b4,b1:b2); %extract pixels contained in bounding box
        
        filter = rand(3);
        I = conv2(double(I_temp),double(filter));
        
        image = bilinearInterpolation(I, [28 28]);

        fullImages(:,:,l) = image(:,:);
        images(:,l) = image(:);
    end

    pos = bbpos(Character);

    %% Propogate image segments though network
    
    set = size(images, 2);
    dimensions = size(images, 1);

    alphabet = 'a':'z';
    numbers = 0:9;
    symbols = {'(',')','=','+','-','div','frac','mul','delta','pm','sqrt','bar','dots','pi','tan','cos','sin'};

    outVecLetters = zeros(length(tempLowerLetters(:,1)), set);
    outVecNumbers = zeros(length(tempDigits(:,1)), set);
    outVecSymbols = zeros(length(tempSymbols(:,1)), set);

    extraImages = zeros(dimensions,set);

    results = zeros(1,3);
    indices = zeros(1,3);

    mergeIndices = zeros(2,3);
    mergeResults =  zeros(2,3);

    mergeResult = zeros(1,2);
    mergeIndex = zeros(1,2);

    class = cell(1,set);
    mergeClass = cell(2,set);

    for i = 1:set
        inputVector = images(:,i);

        [H1, output] = propogateInput(hiddenWeightsLetters, outputWeightsLetters, inputVector);
        outVecLetters(:,i) = output;
        [results(1), indices(1)] = max(output);

        [H1, output] = propogateInput(hiddenWeightsNumbers, outputWeightsNumbers, inputVector);
        outVecNumbers(:,i) = output;
        [results(2), indices(2)] = max(output);

        [H1, output] = propogateInput(hiddenWeightsSymbols, outputWeightsSymbols, inputVector);
        outVecSymbols(:,i) = output;
        [results(3), indices(3)] = max(output);

        [result, index] = max(results);

        if (index == 1)
            class{i} = alphabet(indices(1));
        elseif (index == 2)
            class{i} = num2str(numbers(indices(2)));
        elseif (index == 3)
            class{i} = symbols(indices(3));
        end

        if (i < set)
            tempImage = [fullImages(:,:,i);fullImages(:,:,i+1)];

            tempNewImage = bilinearInterpolation(tempImage, [28 28]);
            extraImages(:,i) = tempNewImage(:);
            inputVector = extraImages(:,i);

            [H1, output] = propogateInput(hiddenWeightsLetters, outputWeightsLetters, inputVector);
            outVecLetters(:,i) = output;
            [ mergeResults(1,1), mergeIndices(1,1)] = max(output);

            [H1, output] = propogateInput(hiddenWeightsNumbers, outputWeightsNumbers, inputVector);
            outVecNumbers(:,i) = output;
            [mergeResults(1,2), mergeIndices(1,2)] = max(output);

            [H1, output] = propogateInput(hiddenWeightsSymbols, outputWeightsSymbols, inputVector);
            outVecSymbols(:,i) = output;
            [mergeResults(1,3), mergeIndices(1,3)] = max(output);

            [mergeResult(1,1), mergeIndex(1,1)] = max(results);

            if (mergeIndex(1,1) == 1)
                mergeClass{1,i} = alphabet(mergeIndices(1,1));
            elseif (mergeIndex(1,1) == 2)
                mergeClass{1,i} = num2str(numbers(mergeIndices(1,2)));
            elseif (mergeIndex(1,1) == 3)
                mergeClass{1,i} = symbols(mergeIndices(1,3));
            end
        end

        if (i < set)
            tempImage = [fullImages(:,:,i) fullImages(:,:,i+1)];
            
            tempNewImage = bilinearInterpolation(tempImage, [28 28]);
            extraImages(:,i) = tempNewImage(:);
            inputVector = extraImages(:,i);

            [H1, output] = propogateInput(hiddenWeightsLetters, outputWeightsLetters, inputVector);
            outVecLetters(:,i) = output;
            [ mergeResults(2,1), mergeIndices(2,1)] = max(output);

            [H1, output] = propogateInput(hiddenWeightsNumbers, outputWeightsNumbers, inputVector);
            outVecNumbers(:,i) = output;
            [mergeResults(2,2), mergeIndices(2,2)] = max(output);

            [H1, output] = propogateInput(hiddenWeightsSymbols, outputWeightsSymbols, inputVector);
            outVecSymbols(:,i) = output;
            [mergeResults(2,3), mergeIndices(2,3)] = max(output);

            [mergeResult(1,2), mergeIndex(1,2)] = max(results);

            if (mergeIndex(1,2) == 1)
                mergeClass{2,i} = alphabet(mergeIndices(2,1));
            elseif (mergeIndex(1,2) == 2)
                mergeClass{2,i} = num2str(numbers(mergeIndices(2,2)));
            elseif (mergeIndex(1,2) == 3)
                mergeClass{2,i} = symbols(mergeIndices(2,3));
            end
        end
    end
end

