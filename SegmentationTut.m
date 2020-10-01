imageDir = 'E:\School\MSU\SemanticSegmentation';
%url = 'http://www.cis.rit.edu/~rmk6217/rit18_data.mat';
%downloadHamlinBeachMSIData(url,imageDir);

load(fullfile(imageDir,'rit18_data','rit18_data.mat'));

train_data = switchChannelsToThirdPlane(train_data);
val_data   = switchChannelsToThirdPlane(val_data);
test_data  = switchChannelsToThirdPlane(test_data);

classNames = [ "RoadMarkings","Tree","Building","Vehicle","Person", ...
               "LifeguardChair","PicnicTable","BlackWoodPanel",...
               "WhiteWoodPanel","OrangeLandingPad","Buoy","Rocks",...
               "LowLevelVegetation","Grass_Lawn","Sand_Beach",...
               "Water_Lake","Water_Pond","Asphalt"]; 
           
cmap = jet(numel(classNames));
B = labeloverlay(histeq(train_data(:,:,4:6)),train_labels,'Transparency',0.8,'Colormap',cmap);

figure
title('Training Labels')
imshow(B)
N = numel(classNames);
ticks = 1/(N*2):1/N:1;
colorbar('TickLabels',cellstr(classNames),'Ticks',ticks,'TickLength',0,'TickLabelInterpreter','none');
colormap(cmap)

save('train_data.mat','train_data');
imwrite(train_labels,'train_labels.png');

% Create an Image Datastore
imds = imageDatastore('train_data.mat','FileExtensions','.mat','ReadFcn',@matReader);

% Create a pixel label Datastore
pixelLabelIds = 1:18;
pxds = pixelLabelDatastore('train_labels.png',classNames,pixelLabelIds);

% Extract 16000, random 256x256 images patches from the training data
% and matching pixel ids
dsTrain = randomPatchExtractionDatastore(imds,pxds,[256,256],'PatchesPerImage',16000);

inputBatch = preview(dsTrain);
disp(inputBatch)

inputTileSize = [256,256,6];
lgraph = createUnet(inputTileSize);
disp(lgraph.Layers)

doTraining = false; 
if doTraining
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    save(['multispectralUnet-' modelDateTime '-Epoch-' num2str(maxEpochs) '.mat'],'net','options');
else 
    load(fullfile(imageDir,'trainedUnet','multispectralUnet.mat'));
end

