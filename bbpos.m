% Finds all positional points on box i
function [bbpos] = bbpos(Character)
    numBoxes = size(Character,2);
    bbpos = zeros(numBoxes,2);
    posinf = zeros(numBoxes,9,2);
    for i = 1:numBoxes
        someBox = Character(i).BoundingBox();
        box = floor(someBox);
        y = box(1);
        x = box(2);
        h = box(3);
        w = box(4);
        
        posinf(i,1,1) = x+floor(w/2); % top
        posinf(i,1,2) = y;
        
        posinf(i,2,1) = x; % top left
        posinf(i,2,2) = y;
        
        posinf(i,3,1) = x; %left
        posinf(i,3,2) = y+floor(h/2);
        
        posinf(i,4,1) = x; % bottom left
        posinf(i,4,2) = y+h;
        
        posinf(i,5,1) = x+floor(w/2); % bottom
        posinf(i,5,2) = y+h;
        
        posinf(i,6,1) = x+w; % bottom right
        posinf(i,6,2) = y+h;
        
        posinf(i,7,1) = x+w; % right
        posinf(i,7,2) = y+floor(h/2);
        
        posinf(i,8,1) = x+w; % top right
        posinf(i,8,2) = y;
        
        posinf(i,9,1) = x+floor(w/2); % center
        posinf(i,9,2) = y+floor(h/2);
        
        if(i > 1)
            c = [posinf(i-1,9,1) posinf(i-1,9,2)];
            c2 = [posinf(i,9,1) posinf(i,9,2)];
            bbpos(i,2) = pdist([c ; c2]);
            
            d = zeros(1,8);
            for l = 1:8
               %calculate distance from bounding box i to center of bounding box i-1 
               t = [posinf(i,l,1) posinf(i,l,1)];
               A = [c ; t];
               d(l) = pdist(A);
               if(l == 1)
                   bbpos(i) = l;
               end
               if(d(bbpos(i,1)) > d(l))
                   bbpos(i,1) = l;
               end
            end
            
        else
            bbpos(i,1) = 8; %start
            bbpos(i,2) = -1;
        end
    end
end