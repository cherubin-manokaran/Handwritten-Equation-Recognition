% Writes image file name and digit to file
function writeEquationsToFile(label, Character, class)
    fName = strcat('predictions.txt');
    id = fopen(fName,'a+');
    
    if (id ~= -1)
        fprintf(id,'%s.png\t%s\n', label, num2str(length(class)));
        
        for i = 1:length(class)
            fprintf(id,'%s\t%s\t%s\t%s\t%s\n', char(class{i}), num2str(round(Character(i).BoundingBox(1))),...
                num2str(round(Character(i).BoundingBox(2))), num2str(round(Character(i).BoundingBox(1)+...
                Character(i).BoundingBox(3))),num2str(round(Character(i).BoundingBox(2)+...
                Character(i).BoundingBox(4))));
        end
        
        fclose(id);
    end
end