%Root folder "directory" for images.
imagefolder='C:\101_ObjectCategories';
imagesetpath = fullfile(imagefolder); 
%Create storage for image data.
images = imageDatastore(imagesetpath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Read image from data store from position n and show it on screen.
%Uncomment if you like to look at.
% n=1;
% img = readimage(images,n);
% imshow(img);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Split data into training and test sets. About 90% of cases belong to training
%set.
[trainingset,testset] = splitEachLabel(images,0.8,'randomized');

%Load in a pre trained deep convolutional neural network.
net = resnet50; %Has 177 layers.
inputSize = net.Layers(1).InputSize; %Size of input data for resnet50.
layer = 'avg_pool'; %Features are extracted after avg_pool layer. Layer 174.
%analyzeNetwork(net); %Present network layer information. Uncomment if you
%like to look at.

%Resizing of trainingset and testset images for resnet50 network.
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingset,'ColorPreprocessing','gray2rgb');
augimdsTest = augmentedImageDatastore(inputSize(1:2),testset,'ColorPreprocessing','gray2rgb');

%Feature extraction of trainingset and testset images using resized images.
%This stage may take some time even with fast computer.
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');

%Labels for each training cases.
labels=trainingset.Labels;
%Unique class labels.
lclasses=unique(labels);


