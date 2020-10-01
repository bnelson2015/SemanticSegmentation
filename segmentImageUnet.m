%  OUT = segmentImageUnet(IM, PATCHSIZE) returns a semantically segmented
%  image, segmented using the network multispectralUnet. The segmentation
%  is performed over each patch of size PATCHSIZE.
%
% Copyright 2019-2020 The MathWorks, Inc.
function out = segmentImageUnet(im, patchSize)

%#codegen

persistent mynet;

if isempty(mynet)
    mynet = coder.loadDeepLearningNetwork('trainedUnet/multispectralUnet.mat');
end

[height, width, nChannel] = size(im);
patch = coder.nullcopy(zeros([patchSize, nChannel-1]));

% pad image to have dimensions as multiples of patchSize
padSize = zeros(1,2);
padSize(1) = patchSize(1) - mod(height, patchSize(1));
padSize(2) = patchSize(2) - mod(width, patchSize(2));

im_pad = padarray (im, padSize, 0, 'post');
[height_pad, width_pad, ~] = size(im_pad);

out = zeros([size(im_pad,1), size(im_pad,2)], 'uint8');

for i = 1:patchSize(1):height_pad    
    for j =1:patchSize(2):width_pad        
        for p = 1:nChannel-1              
            patch(:,:,p) = squeeze( im_pad( i:i+patchSize(1)-1,...
                                            j:j+patchSize(2)-1,...
                                            p));            
        end
         
        % pass in input
        segmentedLabels = activations(mynet, patch, 'Segmentation-Layer');
        
        % Takes the max of each channel (6 total at this point)
        [~,L] = max(segmentedLabels,[],3);
        patch_seg = uint8(L);
        
        % populate section of output
        out(i:i+patchSize(1)-1, j:j+patchSize(2)-1) = patch_seg;
       
    end
end

% Remove the padding
out = out(1:height, 1:width);
