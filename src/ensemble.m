% Setup MatConvNet.
cd('/Users/RachelZheng/Dropbox/EE554/project/Project2/3rd-party/matconvnet/');
run matlab/vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ;

% Obtain and preprocess an image.
pathTrain = '/Users/RachelZheng/Documents/data/EE554_train/';
pathTest = '/Users/RachelZheng/Documents/data/EE554_test/';
imgFiles = dir([pathTrain '*.jpg']);
len = length(imgFiles);
sizeFeature = 1000;
scoreTotal = zeros(len, sizeFeature);

% Adjust the image means
avgImg = zeros(224,224,3);
for i = 1:len
	im = imread([pathTrain imgFiles(i).name]) ;
	im_ = imresize(single(im), net.meta.normalization.imageSize(1:2));
	avgImg = avgImg + im_;
end
avgImg = avgImg/ len;

% getting the scores of the images
for i = 1:len
	im = imread([pathTrain imgFiles(i).name]) ;
	im_ = imresize(single(im), net.meta.normalization.imageSize(1:2)) ;
	im_ = im_ - avgImg;
	res = vl_simplenn(net, im_) ;
	% Show the classification result.
	scoreTotal(i,:) = squeeze(gather(res(end).x)) ;
	disp(['We have the ' num2str(i) 'th image scores']);
end

% training use SVM
labelTrain = [ones(1, 162), 2 * ones(1, 547), 3 * ones(1, 900), 4 * ones(1, 439), 5 * ones(1, 264)];
Mdl = fitcecoc(scoreTotal, labelTrain);

% getting the test scores
imgFiles = dir([pathTest '*.jpg']);
len = length(imgFiles);
scoreTest = zeros(len, sizeFeature);

% getting the scores of the test images
for i = 1:len
	im = imread([pathTest imgFiles(i).name]) ;
	im_ = imresize(single(im), net.meta.normalization.imageSize(1:2)) ;
	im_ = im_ - avgImg;
	res = vl_simplenn(net, im_) ;
	% Show the classification result.
	scoreTest(i,:) = squeeze(gather(res(end).x)) ;
	disp(['We have the ' num2str(i) 'th image scores']);
end

[labelPred, score, cost] = predict(Mdl,scoreTest);
labelTest = [ones(1, 20), 2 * ones(1, 20), 3 * ones(1, 20), 4 * ones(1, 20), 5 * ones(1, 20)];
diff = labelPred - labelTest';
accu = length(diff(find(diff == 0)))/length(diff);

