function [names, labels] = loadEquationImagesFromFolder(folderPath)
    folder = folderPath;
    if ~isdir(folder)
      errorMessage = sprintf('Error: The following folder does not exist:\n%s', folder);
      uiwait(warndlg(errorMessage));
      return;
    end
    
    filePattern = fullfile(folder, '*.png');
    pngFiles = dir(filePattern);
    files = {pngFiles.name};
    names = cell(length(files),1);
    labels = cell(length(files),1);
    
    % Stores all names of files
    for k = 1:length(files)
      fullFileName = fullfile(folder, files{k});
      [~,name,ext] = fileparts(fullFileName);
      labels{k} = name;
      
      filename = strcat(name, ext);
      fullFileName = fullfile(folder, filename);
      names{k} = fullFileName;
    end
end