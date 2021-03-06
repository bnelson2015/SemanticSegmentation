
% Set the image directory
imageDir = 'E:\School\MSU\SemanticSegmentation';
%url = 'http://www.cis.rit.edu/~rmk6217/rit18_data.mat';
%downloadHamlinBeachMSIData(url,imageDir);

% Load downloaded data into MATLAB 
load(fullfile(imageDir,'rit18_data','rit18_data.mat'));

% Move color channels to the third plane
train_data = switchChannelsToThirdPlane(train_data);
val_data   = switchChannelsToThirdPlane(val_data);
test_data  = switchChannelsToThirdPlane(test_data);

% Create a vector of object classes for labeling
classNames = [ "RoadMarkings","Tree","Building","Vehicle","Person", ...
               "LifeguardChair","PicnicTable","BlackWoodPanel",...
               "WhiteWoodPanel","OrangeLandingPad","Buoy","Rocks",...
               "LowLevelVegetation","Grass_Lawn","Sand_Beach",...
               "Water_Lake","Water_Pond","Asphalt"]; 


% Create a colormap for the label data, and overlay class labels on input
% image to show the desired segmentation
cmap = jet(numel(classNames));
B = labeloverlay(histeq(train_data(:,:,4:6)),train_labels,'Transparency',0.8,'Colormap',cmap);

% Display image with hand labeled regions
figure
title('Training Labels')
imshow(B)
N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickLabelInterpreter','none');
colormap(cmap)

% Save training data to a .mat file, save the training labels as a png.
save('train_data.mat','train_data');
imwrite(train_labels,'train_labels.png');

% Create an Image Datastore from training data
imds = imageDatastore('train_data.mat','FileExtensions','.mat','ReadFcn',@matReader);

% Create a pixel label datastore from training lables image
pixelLabelIds = 1:18;
pxds = pixelLabelDatastore('train_labels.png',classNames,pixelLabelIds);

% Extract 16000, random 256x256 images patches from the training data
% and corresponding pixel ids
dsTrain = randomPatchExtractionDatastore(imds,pxds,[256,256],'PatchesPerImage',16000);

% Preview training images
inputBatch = preview(dsTrain);
disp(inputBatch)

% Create U-Net with 256x256x6 input size
inputTileSize = [256,256,6];
lgraph = createUnet(inputTileSize);
disp(lgraph.Layers)

% Train the network if doTraining == true, else use a pretrained network
doTraining = false; 
if doTraining
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    save(['multispectralUnet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net','options');
else 
    load(fullfile(imageDir,'trainedUnet','multispectralUnet.mat'));
end

