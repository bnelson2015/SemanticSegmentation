%% Code Generation for Semantic Segmentation Network by Using U-net
% 
% This example shows code generation for an image segmentation application 
% that uses deep learning. It uses the |codegen| command to generate a MEX 
% function that performs prediction on a DAG Network object
% for U-Net, a deep learning network for image segmentation.
%
% For a similar example covering segmentation of images by using U-Net without
% the |codegen| command, see <matlab:web(fullfile(docroot,'images/multispectral-semantic-segmentation-using-deep-learning.html'))
% Semantic Segmentation of Multispectral Images Using Deep Learning>.
%
% Copyright 2018-2019 The MathWorks, Inc.
%% Prerequisites
% * CUDA(R) enabled NVIDIA(R) GPU with compute capability 3.2 or higher.
% * NVIDIA CUDA toolkit and driver.
% * NVIDIA cuDNN library.
% * Environment variables for the compilers and libraries. For information 
% on the supported versions of the compilers and libraries, see 
% <docid:gpucoder_gs#mw_aa8b0a39-45ea-4295-b244-52d6e6907bff
% Third-party Products>. For setting up the environment variables, see 
% <docid:gpucoder_gs#mw_453fbbd7-fd08-44a8-9113-a132ed383275
% Environment Variables>.
% * GPU Coder(TM) Interface for Deep Learning Libraries support package. To
% install this support package, use the
% <matlab:matlab.addons.supportpackage.internal.explorer.showSupportPackages('GPU_DEEPLEARNING_LIB','tripwire')
% Add-On Explorer>.
%% Verify GPU Environment
%
% Use the <docid:gpucoder_ref#mw_0957d820-192f-400a-8045-0bb746a75278 coder.checkGpuInstall> function
% to verify that the compilers and libraries necessary for running this example
% are set up correctly.
envCfg = coder.gpuEnvConfig('host');
envCfg.DeepLibTarget = 'cudnn';
envCfg.DeepCodegen = 1;
envCfg.Quiet = 1;
coder.checkGpuInstall(envCfg);

%% Segmentation Network
% 
% U-Net [1] is a type of convolutional neural network (CNN)
% designed for semantic image segmentation. In U-Net, the initial series of
% convolutional layers are interspersed with max pooling layers,
% successively decreasing the resolution of the input image. These layers
% are followed by a series of convolutional layers interspersed with
% upsampling operators, successively increasing the resolution of the input
% image. Combining these two series paths forms a U-shaped graph. The 
% network was originally trained for and used to perform
% prediction on biomedical image segmentation applications. This
% example demonstrates the ability of the network to track changes in
% forest cover over time. Environmental agencies track deforestation 
% to assess and qualify the environmental and ecological health of a region.
% 
% Deep-learning-based semantic segmentation can yield a precise measurement of 
% vegetation cover from high-resolution aerial photographs. One challenge is 
% differentiating classes that have similar visual characteristics, such as trying 
% to classify a green pixel as grass, shrubbery, or tree. To increase classification 
% accuracy, some data sets contain multispectral images that provide additional 
% information about each pixel. For example, the Hamlin Beach State Park data 
% set supplements the color images with near-infrared channels that provide a 
% clearer separation of the classes.
%
% This example uses the Hamlin Beach State Park Data [2] along with a
% pretrained U-Net network in order to correctly classify each pixel.
%
% The U-Net used is trained to segment pixels belonging to 18 classes which 
% includes: 
%
%  0. Other Class/Image Border      7. Picnic Table         14. Grass
%  1. Road Markings                 8. Black Wood Panel     15. Sand
%  2. Tree                          9. White Wood Panel     16. Water (Lake)
%  3. Building                     10. Orange Landing Pad   17. Water (Pond)
%  4. Vehicle (Car, Truck, or Bus) 11. Water Buoy           18. Asphalt (Parking Lot/Walkway)
%  5. Person                       12. Rocks
%  6. Lifeguard Chair              13. Other Vegetation

%% The |segmentImageUnet| Entry-Point Function
%
% The
% <matlab:edit(fullfile(matlabroot,'examples','deeplearning_shared','main','segmentImageUnet.m'))
% segmentImageUnet.m> entry-point function performs patchwise semantic segmentation on the input
%  image by using the multispectralUnet network found in the
%  |multispectralUnet.mat| file. The function loads the network object from
%  the |multispectralUnet.mat| file into a persistent variable _mynet_ and 
% reuses the persistent variable on subsequent prediction calls.
%
type('segmentImageUnet.m')

%% Get Pretrained U-Net DAG Network Object
%
trainedUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/multispectralUnet.mat';
downloadTrainedUnet(trainedUnet_url,pwd);

%%
%
ld = load("trainedUnet/multispectralUnet.mat");
net = ld.net;

%%
% The DAG network contains 58 layers including convolution, max pooling, depth 
% concatenation, and the pixel classification output layers. To display an 
% interactive visualization of the deep learning network architecture, use the 
% <docid:nnet_ref#mw_8d52b67d-b6b3-4c62-b573-56fdf4dce6a0 analyzeNetwork> 
% function.
%   analyzeNetwork(net);

%% Prepare Data
%
% Download the Hamlin Beach State Park data.
if ~exist(fullfile(pwd,'data'))
    url = 'http://www.cis.rit.edu/~rmk6217/rit18_data.mat';
    downloadHamlinBeachMSIData(url,pwd+"/data/");
end

%%
%
% Load and examine the data in MATLAB.
load(fullfile(pwd,'data','rit18_data','rit18_data.mat'));

% Examine data
whos test_data

%%
% The image has seven channels. The RGB color channels are
% the fourth, fifth, and sixth image channels. The first three channels
% correspond to the near-infrared bands and highlight different components
% of the image based on their heat signatures. Channel 7 is a mask that
% indicates the valid segmentation region.
%
% The multispectral image data is arranged as
% numChannels-by-width-by-height arrays. In MATLAB, multichannel
% images are arranged as width-by-height-by-numChannels arrays. To reshape
% the data so that the channels are in the third dimension, use the helper
% function, |switchChannelsToThirdPlane|.
test_data  = switchChannelsToThirdPlane(test_data);

% Confirm data has the correct structure (channels last).
whos test_data


%% Run MEX Code Generation 
%
% To generate CUDA code for
% <matlab:edit(fullfile(matlabroot,'examples','deeplearning_shared','main','segmentImageUnet.m')) segmentImageUnet.m> 
% entry-point function, create a GPU Configuration object for a MEX target 
% setting the target language to C++. Use the
% <docid:gpucoder_ref#mw_e8e85f8e-8dde-45b6-9ec5-f121a79dc48f coder.DeepLearningConfig>
% function to create a |CuDNN| deep learning configuration object and
% assign it to the |DeepLearningConfig| property of the GPU code
% configuration object. Run the |codegen| command specifying an input size of
% [12446,7654,7] and a patch size of [1024,1024]. These
% values correspond to the entire test_data size. The smaller patch sizes
% speed up inference. To see how the patches are calculated, see the 
% |segmentImageUnet.m| entry-point function.
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
codegen -config cfg segmentImageUnet -args {ones(size(test_data),'uint16'),coder.Constant([1024 1024])} -report

%% Run Generated MEX to Predict Results for test_data
% This |segmentImageUnet| function takes in the data to test (test_data)
% and a vector containing the dimensions of the patch size to use. Take 
% patches of the image, predict the pixels in a particular patch, then 
% combine all the patches together. Due to the size of test_data 
% (12446x7654x7), it is easier to process such a large image in patches.
segmentedImage = segmentImageUnet_mex(test_data,[1024 1024]);

%%
% To extract only the valid portion of the segmentation, multiply the
% segmented image by the mask channel of the test data.
segmentedImage = uint8(test_data(:,:,7)~=0) .* segmentedImage;

%%
% Because the output of the semantic segmentation is noisy, remove the noise
% and stray pixels by using the |medfilt2| function.
segmentedImage = medfilt2(segmentedImage,[5,5]);

%% Display U-Net Segmented test_data 
% The following line of code creates a vector of the class names.
classNames = [ "RoadMarkings","Tree","Building","Vehicle","Person", ...
               "LifeguardChair","PicnicTable","BlackWoodPanel",...
               "WhiteWoodPanel","OrangeLandingPad","Buoy","Rocks",...
               "LowLevelVegetation","Grass_Lawn","Sand_Beach",...
               "Water_Lake","Water_Pond","Asphalt"]; 

%%
% Overlay the labels on the segmented RGB test image and add a color bar to the
% segmentation image.
cmap = jet(numel(classNames));
B = labeloverlay(imadjust(test_data(:,:,4:6),[0 0.6],[0.1 0.9],0.55),segmentedImage,'Transparency',0.8,'Colormap',cmap);
figure
imshow(B)

N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickLabelInterpreter','none');
colormap(cmap)
title('Segmented Image');

%% References
% [1] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net:
% Convolutional Networks for Biomedical Image Segmentation." _arXiv
% preprint arXiv:1505.04597,_ 2015.
%
% [2] Kemker, R., C. Salvaggio, and C. Kanan. "High-Resolution
% Multispectral Dataset for Semantic Segmentation." CoRR, abs/1703.01918,
% 2017.