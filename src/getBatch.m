function [im, labels] = getBatch(imdb,batch)
% for i = 1:3
    averageImage(:,:,1) = 129.3348;
    averageImage(:,:,2) = 123.2025;
    averageImage(:,:,3) = 114.4834;
% end
im = imdb.images.data(:,:,:,batch);
im = single(im) ; % note: 255 range
im = imresize(im, [224 224]) ;
im = bsxfun(@minus,im,averageImage) ;
labels = imdb.images.label(1,batch) ;

