function [Character, indices] = segmenter(varargin)

    % varargin = 'C:\Users\Manokaran\Documents\MATLAB\annotated\SKMBT_36317040717363_eq33.png';
    nVarargs = length(varargin);

    Im = imread(varargin{1});

    Im_saved = Im;

    %Make binary
    Im(Im < 100) = 0;
    Im(Im >= 100) = 1;

    %Segment out all connected regions
    ImL = bwlabel(Im); 

    %Get labels for all distinct regions
    labels = unique(ImL);

    %Remove label 0, corresponding to background
    labels(labels==0) = [];

    %Get bounding box for each segmentation
    Character = struct('BoundingBox',zeros(1,4));
    nrValidDetections = 0;
    for i=1:length(labels)
        D = regionprops(ImL==labels(i));
        if (D.Area > 100)
            nrValidDetections = nrValidDetections + 1;
            Character(nrValidDetections).BoundingBox = D.BoundingBox;
        end 
    end
    
    indices = cell(nrValidDetections,1);
    
    patternTrue = strfind(varargin, 'eq12.png');
    
    if (length(patternTrue) >= 1)
%         figure;
%         colormap('gray') 
%         imagesc(Im_saved); 
        
        for i=1:nrValidDetections
%             rectangle('Position',[Character(i).BoundingBox(1) ...
%                                   Character(i).BoundingBox(2) ...
%                                   Character(i).BoundingBox(3) ...
%                                   Character(i).BoundingBox(4)],...
%                                   'EdgeColor','r', 'LineWidth', 3);

            indices{i,1} = sprintf('%s\t%s\t%s\t%s\t',num2str(Character(i).BoundingBox(1)), num2str(Character(i).BoundingBox(2)),...
                num2str(Character(i).BoundingBox(3)), num2str(Character(i).BoundingBox(4)));

        end 
    end
end
