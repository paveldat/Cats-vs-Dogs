%% load data, split train:validate as 7:3

rng("default") % default seed for reproducibility 

data = imageDatastore("petImagesSelected","IncludeSubfolders", true, "LabelSource","foldernames");
data.shuffle(); %перемешивает
[dataTrain, dataValidate] = splitEachLabel(data, .7); % разбивает датасет на 2 сета
%% explore the dataset

namedDatasets = {
    "train", dataTrain; 
    "validate", dataValidate};

for i = 1:size(namedDatasets,1)
    title = namedDatasets{i,1};
    dataSubset = namedDatasets{i,2};

    disp(title + ": sample size: " + numel(dataSubset.Files));
    disp(title + ": labels distribution: ")
    disp(dataSubset.countEachLabel())
end

%% look at the pre-trained convolutional neural network

nnet = vgg19;
analyzeNetwork(nnet);
%% set up the new network layers (all pre-trained except for two)

nClasses = numel(categories(data.Labels));
lgraph = layerGraph(nnet);
lgraph = replaceLayer(lgraph,lgraph.Layers(end-2).Name, ...
    fullyConnectedLayer(nClasses, ...
        "Name", "fine_tune_fc", ...
        "WeightLearnRateFactor", 10, ...
        "BiasLearnRateFactor", 10));
lgraph = replaceLayer(lgraph,lgraph.Layers(end).Name, ...
    classificationLayer("Name", "fine_tune_output"));
%% update layers properties
layers = lgraph.Layers;
connections = lgraph.Connections;

for ii = 1:10
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph,layers(i));
end
for c = 1:size(connections,1)
    lgraph = connectLayers(lgraph,connections.Source{c},connections.Destination{c});
end

%% set up data augmenter

augmenter = imageDataAugmenter( ...
    "RandXReflection", true, ...
    "RandRotation", [-45, 45]);

inputDims = layers(1).InputSize(1:2);
augTrain = augmentedImageDatastore(inputDims,dataTrain, 'DataAugmentation',augmenter, "ColorPreprocessing","gray2rgb");
augValidate = augmentedImageDatastore(inputDims,dataValidate, "ColorPreprocessing","gray2rgb");

%% set up optimization parameters
miniBatchSize = 22;
options = trainingOptions("sgdm", ...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.001,...
    "MaxEpochs", 16,...
    "MiniBatchSize", 22, ...
    "Shuffle","every-epoch", ...
    "ValidationData",augValidate, ...
    "ValidationFrequency", 10, ...
    "Verbose",false, ...
    "Plots","training-progress");

%% train the net
nnetTransfer = trainNetwork(augTrain,lgraph,options);

%% evaluate the net

[yPredicted,scores] = classify(nnetTransfer,augValidate);
yGroundTruth = dataValidate.Labels;

figure('Units','normalized','Position',[0.2 0.2 0.4 0.4]);
cm = confusionchart(yGroundTruth,yPredicted);
cm.Title = 'Confusion matrix for validation data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

%%
myData = imageDatastore("myPictures", "IncludeSubfolders", true, "LabelSource","foldernames");
myAugmented = augmentedImageDatastore(inputDims,myData);
classes = classify(nnetTransfer, myAugmented);
for i = 1:numel(classes)
    disp(myAugmented.Files(i) + ": " + string(classes(i)));
end
