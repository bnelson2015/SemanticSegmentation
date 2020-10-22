function lgraph = testUnet(imageInputTiles)

sections = 4;

layers = imageInputLayer(imageInputTiles, 'Name', 'Input Layer');


% Create Encoder Layers
for i = 1:sections
    
    if i == 1
        
        conv1 = convolution2dLayer([3,3], 64*2^(i-1), 'Padding', 1, ...
            'NumChannels', 3, 'Name', ['E-', num2str(i),'-Conv1']);
        
        relu1 = reluLayer('Name', ['E-', num2str(i),'-ReLU1'])
        
        conv2 = convolution2dLayer([3,3], 64*2^(i-1), 'Padding', 1,...
            'Name', ['E-', num2str(i),'-Conv2']);
        
        relu2 = reluLayer('Name', ['E-', num2str(i),'-ReLU2'])
        
        pool = maxPooling2dLayer([2,2], 'Stride', 2, 'Name', ['E-', num2str(i),'-MaxPool']);
        
        layers = [layers; conv1; relu1; conv2; relu2; pool];
        
    else
        
        conv1 = convolution2dLayer([3,3], 64*2^(i-1), 'Padding', 1, ...
            'Name', ['E-', num2str(i),'-Conv1']);
        
        relu1 = reluLayer('Name', ['E-', num2str(i),'-ReLU1'])
        
        conv2 = convolution2dLayer([3,3], 64*2^(i-1), 'Padding', 1,...
            'Name', ['E-', num2str(i),'-Conv2']);
        
        relu2 = reluLayer('Name', ['E-', num2str(i),'-ReLU2'])
        
        pool = maxPooling2dLayer([2,2], 'Stride', 2, 'Name', ['E-', num2str(i),'-MaxPool']);
        
        if i==4
            dropOut = dropoutLayer(0.5, 'Name', 'E-DropoutLayer');
            layers = [layers; conv1; relu1; conv2; relu2; dropOut; pool];
            
        else       
            layers = [layers; conv1; relu1; conv2; relu2; pool];
        end
    end
end

% Create Middle Layers

        conv1 = convolution2dLayer([3,3], 64*2^(4), 'Padding', 1, ...
            'Name', ['M-Conv1']);
        
        relu1 = reluLayer('Name', ['M-ReLU1'])
        
        conv2 = convolution2dLayer([3,3], 64*2^(4), 'Padding', 1,...
            'Name', ['M-Conv2']);
        
        relu2 = reluLayer('Name', ['M-ReLU2'])
        
        dropOut = dropoutLayer(0.5, 'Name', 'M-DropoutLayer');

       layers = [layers; conv1; relu1; conv2; relu2; dropOut]

% Create Decoder Layers
for i = 1:sections
    
    upconv1 = transposedConv2dLayer([2,2], 512/2^(i-1), 'Stride', 2, 'Name', ['D-', num2str(i),'-UpConv1']);
    
    uprelu1 = reluLayer('Name',['D-', num2str(i),'-UpReLU1'])
    
    depthCat = depthConcatenationLayer(2,'Name',...
        ['D-', num2str(i),'-DepthCat']);
    
    conv1 = convolution2dLayer([3,3], 512/2^(i-1), 'Padding', 1, ...
        'Name', ['D-', num2str(i),'-Conv1']);

    relu1 = reluLayer('Name', ['D-', num2str(i),'-ReLU1'])

    conv2 = convolution2dLayer([3,3], 512/2^(i-1), 'Padding', 1,...
        'Name', ['D-', num2str(i),'-Conv2']);

    relu2 = reluLayer('Name', ['D-', num2str(i),'-ReLU2'])
    
    layers = [layers; upconv1; uprelu1; depthCat; conv1; relu1; conv2; relu2]
        
end

finalConv = convolution2dLayer([1, 1], 1, 'Name', 'FinalConv');

sm = softmaxLayer('Name', 'SoftMax');

pixelClass = pixelClassificationLayer('Name', 'PixelClassification');

layers = [layers; finalConv; sm; pixelClass];

lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph, 'E-1-ReLU2', 'D-4-DepthCat/in2');
lgraph = connectLayers(lgraph, 'E-2-ReLU2', 'D-3-DepthCat/in2');
lgraph = connectLayers(lgraph, 'E-3-ReLU2', 'D-2-DepthCat/in2');
lgraph = connectLayers(lgraph, 'E-4-ReLU2', 'D-1-DepthCat/in2');

analyzeNetwork(lgraph)

end

