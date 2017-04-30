function [net, info] = finetune(varargin)
% finetune the existing network for the new data
usrs = {'xvz5220-admin', 'RachelZheng'};
usr = usrs{2};
%------------------
% setup the environment 
%------------------
% add the matconvnet directory
addpath(genpath(['/Users/' usr '/Dropbox/EE554/project/Project2/3rd-party/matconvnet/']));
vl_setupnn;
opts.batchNormalization = false ;
opts.train = struct() ;
while length(varargin) > 0,	[opts, varargin] = vl_argparse(opts, varargin) ; end
% setup the directory
opts.dataDir = ['/Users/' usr '/Documents/data/EE554_train2/'];
opts.expDir = ['/Users/' usr '/Documents/data/EE554_test2/'];
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;
% load the pre-trained model
opts.modelPath = ['/Users/' usr '/Dropbox/EE554/project/Project2/3rd-party/matconvnet/imagenet-vgg-f.mat'];
load(opts.modelPath);
opts.train = struct() ;

%------------------
% load the dataset
%------------------
imgFiles = dir([opts.dataDir '*.jpg']);
len = length(imgFiles);
scoreTotal = zeros(len, 4096);
for i = 1:len
	im = imread([opts.dataDir imgFiles(i).name]) ;
	im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
	im_ = im_ - net.meta.normalization.averageImage;
	res = vl_simplenn(net, im_) ;
	featureScore(i, :) = squeeze(gather(res(end - 2).x)) ;
	disp(['We have the ' num2str(i) 'th image scores']);
end
% save the scores
save('trainScoreNew.mat','featureScore');

% load the testing set
trainScore = featureScore;
imgFiles = dir([opts.expDir '*.jpg']);
len = length(imgFiles);
scoreTotal = zeros(len, 4096);
for i = 1:len
	im = imread([opts.expDir imgFiles(i).name]) ;
	im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
	im_ = im_ - net.meta.normalization.averageImage;
	res = vl_simplenn(net, im_) ;
	featureScore(i, :) = squeeze(gather(res(end - 2).x)) ;
	disp(['We have the ' num2str(i) 'th image scores']);
end
save('testScoreNew.mat','featureScore');

% load the training labels
labelTrain = [zeros(1, 957), ones(1, 1062)];
labelTest = [zeros(1, 20), ones(1, 20)];

%----------------------
% construct the fully-connected layer
%----------------------
hiddenLayerSize = 2;
net = patternnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainParam.epochs=10000;
net.trainParam.lr=0.1;
net.trainParam.goal= 10^(-6);
[net,tr] = train(net,trainScore',labelTrain);
outputs = round(net(trainScore'));
outputsTest = round(net(testScore'));
errors = gsubtract(labelTrain, outputs);
errorsTest = gsubtract(labelTest, outputsTest);
performance = perform(net, labelTrain, outputs);

% View the Network
view(net)
