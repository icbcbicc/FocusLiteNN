disp("This Matlab version is only for the purpose of exactly reproducing the results of the PyTorch version,
such that the efficiency of this Matlab code is not taken into condiseration.")

clear;
clc;

load("matlab/FocusLiteNN1channel.mat");

im_name = "imgs/TCGA@Focus_patch_i_9651_j_81514.png";
im = im2double(imread(im_name));

patches = denseSample(im, 235, 128, 0);
window_patches = denseSample(patches, 7, 5, 1);
[sample, num_patches, window, window, channel] = size(window_patches);

score_list = zeros(sample, num_patches);

for i=1:sample
    for j=1:num_patches
        patch = window_patches(i, j, :, :, :);
        patch = reshape(patch, [7,7,3]);
        score_list(i, j) = sum(patch .* conv_weight, 'all') + conv_bias;
    end
end

final_score = mean(min(score_list, [], 2));

fprintf("Image: %s\t score: %.4f\n", im_name, final_score)

function patches = denseSample(im, window, stride, padding)
    if length(size(im)) == 3
        [height, width, channel] = size(im);
        num_patches_h = floor((2*padding + height - window) / stride + 1);
        num_patches_w = floor((2*padding + width - window) / stride + 1);
        num_patches = num_patches_h * num_patches_w;

        patches = zeros(num_patches, window, window, channel);
        for h=1:num_patches_h
            for w=1:num_patches_w
                patches((h-1)*num_patches_w + w, :, :, :) = im(1+(h-1)*stride:window+(h-1)*stride, 1+(w-1)*stride:window+(w-1)*stride, :);
            end
        end
    elseif length(size(im)) == 4
        [sample, height, width, channel] = size(im);
        num_patches_h = floor((2*padding + height - window) / stride + 1);
        num_patches_w = floor((2*padding + width - window) / stride + 1);
        num_patches = num_patches_h * num_patches_w;

        patches = zeros(sample, num_patches, window, window, channel);
        im = padarray(im, [0, padding, padding, 0], 0);
        for i=1:sample
            for h=1:num_patches_h
                for w=1:num_patches_w
                    patches(i, (h-1)*num_patches_w + w, :, :, :) = im(i, 1+(h-1)*stride:window+(h-1)*stride, 1+(w-1)*stride:window+(w-1)*stride, :);
                end
            end
        end
    end
end
