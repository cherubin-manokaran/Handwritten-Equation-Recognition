% Preprocess images using bilinear interpolation
function output = bilinearInterpolation(input_image, output_dims)
    input_rows = size(input_image,1);
    input_cols = size(input_image,2);
    output_rows = output_dims(1);
    output_cols = output_dims(2);
       
    S_R = input_rows / output_rows;
    S_C = input_cols / output_cols;

    % Define grid of co-ordinates in our image
    % Generate (x,y) pairs for each point in our image
    [cf, rf] = meshgrid(1 : output_cols, 1 : output_rows);

    rf = rf * S_R;
    cf = cf * S_C;

    r = floor(rf);
    c = floor(cf);

    % Any values out of range, cap
    r(r < 1) = 1;
    c(c < 1) = 1;
    r(r > input_rows - 1) = input_rows - 1;
    c(c > input_cols - 1) = input_cols - 1;

    % Let delta_R = rf - r and delta_C = cf - c
    delta_R = rf - r;
    delta_C = cf - c;

    % Get column major indices for each point we wish
    in1_ind = sub2ind([input_rows, input_cols], r, c);
    in2_ind = sub2ind([input_rows, input_cols], r+1,c);
    in3_ind = sub2ind([input_rows, input_cols], r, c+1);
    in4_ind = sub2ind([input_rows, input_cols], r+1, c+1);       

    % Go through each channel for the case of colour
    % Create output image that is the same class as input
    output = zeros(output_rows, output_cols, size(input_image, 3));
    output = cast(output, 'like', input_image);

    for idx = 1 : size(input_image, 3)
        chan = double(input_image(:,:,idx));
        
        % Interpolate the channel
        tmp = chan(in1_ind).*(1 - delta_R).*(1 - delta_C) + ...
                       chan(in2_ind).*(delta_R).*(1 - delta_C) + ...
                       chan(in3_ind).*(1 - delta_R).*(delta_C) + ...
                       chan(in4_ind).*(delta_R).*(delta_C);
        output(:,:,idx) = cast(tmp, 'like', input_image);
    end
end