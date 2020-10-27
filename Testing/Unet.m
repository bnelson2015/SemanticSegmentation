function lgraph = Unet(inputImageSize)

    layers = imageInputLayer(inputImageSize, 'Name', 'InputLayer');
    
    %----- Encoder -----% 
    
    % Encode 1
    conv11 = convolution2dLayer([3, 3], 64, 'Padding', [1 1], 'NumChannels',...
        3, 'Name', '1');
    
    relu11 = reluLayer('Name', '2');
    
    conv12 = convolution2dLayer([3, 3], 64, 'Padding', [1 1], 'Name', '3');
    
    relu12 = reluLayer('Name', 'E1');
    
    pool1 = maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', '4');
    
    layers = [layers; conv11; relu11; conv12; relu12; pool1;];
    
    % Encode 2
    conv21 = convolution2dLayer([3, 3], 128, 'Padding', [1 1], 'Name', '5')
    
    relu21 = reluLayer('Name', '6');
    
    conv22 = convolution2dLayer([3, 3], 128, 'Padding', [1 1],'Name', '7');
    
    relu22 = reluLayer('Name', 'E2');
    
    pool2 = maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', '9');
    
    layers = [layers; conv21; relu21; conv22; relu22; pool2;];
    
    % Encode 3
    conv31 = convolution2dLayer([3, 3], 256, 'Padding', [1 1], 'Name', '10');
    
    relu31 = reluLayer('Name', '11');
    
    conv32 = convolution2dLayer([3, 3], 256, 'Padding', [1 1], 'Name', '12');
    
    relu32 = reluLayer('Name', 'E3');
    
    pool3 = maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', '14');
    
    layers = [layers; conv31; relu31; conv32; relu32; pool3;];
    
    % Encode 4
    conv41 = convolution2dLayer([3, 3], 512, 'Padding', [1 1], 'Name', '15');
    
    relu41 = reluLayer('Name', '16');
    
    conv42 = convolution2dLayer([3, 3], 512, 'Padding', [1 1], 'Name', '17');
    
    relu42 = reluLayer('Name', '18');
    
    pool4 = maxPooling2dLayer([2, 2], 'Stride', 2, 'Name', '19');
    
    dropout1 = dropoutLayer(0.5, 'Name', 'E4');
    
    layers = [layers; conv41; relu41; conv42; relu42; pool4; dropout1];
    
    
    %----- Middle -----% 
    
    convm1 = convolution2dLayer([3, 3], 1024, 'Padding', [1 1], 'Name', '21');
    
    relum1 = reluLayer('Name', '22');
    
    convm2 = convolution2dLayer([3, 3], 1024, 'Padding', [1 1], 'Name', '23');
    
    relum2 = reluLayer('Name', '24');
    
    layers = [layers; convm1; relum1; convm2; relum2];
    
    
    %----- Decode -----%
    
    deconv1 = transposedConv2dLayer([3,3], 512, 'Stride', 1, 'Name', '25', 'cropping', 1);
    
    upRelu1 = reluLayer('Name', 'UR1');
    
    depthCat1 = depthConcatenationLayer(2, 'Name', 'D1');
    
    dropoutd1 = dropoutLayer(0.5, 'Name', '28');
    
    convd11 = convolution2dLayer([3, 3], 512, 'Padding', [1 1], 'Name', '29');
    
    relud11 = reluLayer('Name', '30');
    
    convd12 = convolution2dLayer([3, 3], 512, 'Padding', [1 1], 'Name', '31');
    
    relud12 = reluLayer('Name', '32');
    
    layers = [layers; deconv1; upRelu1; depthCat1; dropoutd1; convd11; relud11; ...
         convd12; relud12];
     
    % Decode 2
    
    deconv2 = transposedConv2dLayer([1,1], 256, 'Stride', 1, 'Name', '33');
    
    upRelu2 = reluLayer('Name', 'UR2');
    
    depthCat2 = depthConcatenationLayer(2, 'Name', 'D2');
    
    dropoutd2 = dropoutLayer(0.5, 'Name', '34');
    
    convd21 = convolution2dLayer([3, 3], 256, 'Padding', [1 1], 'Name', '35');
    
    relud21 = reluLayer( 'Name', '36');
    
    convd22 = convolution2dLayer([3, 3], 256, 'Padding', [1 1], 'Name', '37');
    
    relud22 = reluLayer( 'Name', '38');
    
    layers = [layers; deconv2; upRelu2; depthCat2; dropoutd2; convd21; relud21; ...
         convd22; relud22];
     
     % Decode 3
    
    deconv3 = transposedConv2dLayer([3,3], 128, 'Stride', 2, 'Name', '39');
    
    upRelu3 = reluLayer('Name', 'UR3');
    
    depthCat3 = depthConcatenationLayer(2, 'Name', 'D3');
    
    dropoutd3 = dropoutLayer(0.5, 'Name', '41');
    
    convd31 = convolution2dLayer([3, 3], 128, 'Padding', [1 1], 'Name', '42');
    
    relud31 = reluLayer('Name', '43');
    
    convd32 = convolution2dLayer([3, 3], 128, 'Padding', [1 1], 'Name', '44');
    
    relud32 = reluLayer('Name', '45');
    
    layers = [layers; deconv3; upRelu3; depthCat3; dropoutd3; convd31; relud31; ...
         convd32; relud32];
     
     % Decode 4
     
    deconv4 = transposedConv2dLayer([3,3], 64, 'Stride', 2, 'Name', '46');
    
    upRelu4 = reluLayer('Name', 'UR4');
    
    depthCat4 = depthConcatenationLayer(2, 'Name', 'D4');
    
    dropoutd4 = dropoutLayer(0.5, 'Name', '48');
    
    convd41 = convolution2dLayer([3, 3], 64, 'Padding', [1 1], 'Name', '49');
    
    relud41 = reluLayer('Name', '50');
    
    convd42 = convolution2dLayer([3, 3], 64, 'Padding', [1 1], 'Name', '51');
    
    relud42 = reluLayer('Name', '52');
    
    layers = [layers; deconv4; upRelu4; depthCat4; dropoutd4; convd41; relud41; ...
         convd42; relud42];
     
    % Final
    
    finalcon = convolution2dLayer([1, 1], 1, 'Name', '53');
    
    sm = softmaxLayer('Name', '54');
    
    pixelC = pixelClassificationLayer('Name', '55');
    
    layers = [layers; finalcon; sm; pixelC];
    
    lgraph = layerGraph(layers);
    
    lgraph = connectLayers(lgraph, 'E1', 'D4/in2')
    lgraph = connectLayers(lgraph, 'E2', 'D3/in2')
    lgraph = connectLayers(lgraph, 'E3', 'D2/in2')
    lgraph = connectLayers(lgraph, 'E4', 'D1/in2')
    
    analyzeNetwork(lgraph)
    
end

