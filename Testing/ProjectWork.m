
% Create a Tiff object and pull image information
tifLink = Tiff('#34 a7-2 20x.tif', 'r');
InfoImage = imfinfo('#34 a7-2 20x.tif');
mImage = InfoImage(1).Width;
nImage = InfoImage(1).Height;
NumberImages = length(InfoImage);

% Image layer storage
FullImage = zeros(nImage, mImage, NumberImages, 'uint16');

% Store layers of the tiff in the FullImage array
for i = 1:NumberImages
    tifLink.setDirectory(i);
    FullImage(:,:,i) = tifLink.read();
end
tifLink.close();

% Display each layer of the tiff with scaled colors
figure; title('Are the Channels correct?'); subplot(1,3,1); imagesc(FullImage(:,:,1));
subplot(1,3,2); imagesc(FullImage(:,:,2));
subplot(1,3,3); imagesc(FullImage(:,:,3));

% Label each layer as the E, D, or N channel
x = inputdlg({'Epethelial Channel','Dendridic Channel','Nuclei Channel'},...
          'Input Channels', [1 20; 1 20; 1 20]); 

% Put the images in the correct order (E, D, N)
Shuffled = cat(3,FullImage(:,:,str2num(x{1})),FullImage(:,:,str2num(x{2})),FullImage(:,:,str2num(x{3})));

figure; title('Shuffled Channels'); subplot(1,3,1); imagesc(Shuffled(:,:,1));
subplot(1,3,2); imagesc(Shuffled(:,:,2));
subplot(1,3,3); imagesc(Shuffled(:,:,3));

% Create the logical array of ROIs
ROIs = selection_logical('#34 a7-2 20x.tif ROI.csv');

%padd ones if size doesn't match and error is thrown
ROIs = padarray(ROIs',[abs(size(FullImage(:,:,str2num(x{3})),2)-size(ROIs,2)) 2],1,'post')';
ROIs = padarray(ROIs',[abs(size(FullImage(:,:,str2num(x{3})),1)-size(ROIs,1)) 1],1,'post')';

% adjust size if ROI doesn't match image size
ROIs = ROIs(1:size(FullImage(:,:,str2num(x{3})),1),1:size(FullImage(:,:,str2num(x{3})),2));

figure; imagesc(ROIs);

classNames = ['ND', 'D'];
pixelLabels = 0:1;
imwrite(FullImage, 'test.png')
imwrite(ROIs, 'training_labels.png')

imds = imageDatastore('test_enhanced.png', 'ReadFcn', @pngRead); 

pxds = pixelLabelDatastore('training_labels.png', classNames, pixelLabels);

dsTrain = randomPatchExtractionDatastore(imds,pxds,[64,64],'PatchesPerImage',100);

inputBatch = preview(dsTrain);
disp(inputBatch);

inputTileSize = [64, 64, 3];
lgraph = testUnet(inputTileSize);
disp(lgraph.Layers);

initialLearningRate = 0.05;
maxEpochs = 15;
minibatchSize = 16;
l2reg = 0.0001;

options = trainingOptions('sgdm',...
    'InitialLearnRate',initialLearningRate, ...
    'Momentum',0.9,...
    'L2Regularization',l2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',minibatchSize,...
    'LearnRateSchedule','piecewise',...    
    'Shuffle','every-epoch',...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',0.05, ...
    'Plots','training-progress', ...
    'VerboseFrequency',20);

modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
[net,info] = trainNetwork(dsTrain,lgraph,options);