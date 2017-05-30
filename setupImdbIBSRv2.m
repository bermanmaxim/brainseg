function imdb = setupImdbIBSRv2(net,indsTrain,indsVal,augment,view, cropped)
% -------------------------------------------------------------------------
if nargin < 2, indsTrain = 1:5; end  % indices of the train examples
if nargin < 3, indsVal   = [];  end  % indices of the validation examples
if nargin < 4, augment   = true;end  % augment data using shifts
if nargin < 5, view      = 1;   end  % volume view used (axial, coronal, sagital)
if nargin < 6, cropped   = true;end  % crop isbr images first

% Read original patches
switch view
    case {1,'axial'}
        permvec = [1 2 3];
    case {2, 'sagittal'} 
        permvec = [2 3 1];
    case {3, 'coronal'} 
        permvec = [3 1 2];
    otherwise
        error('Invalid view')
end
paths    = setPaths();
dirs     = dir(paths.IBSR); 
dirs = dirs(~ismember({dirs.name},{'.','..'}));
dirs = dirs([dirs.isdir]); % only look at dirs
nFiles   = numel(dirs); assert(nFiles == 18);
insz     = [256,128,256]; 
insz     = insz(permvec); nChannelsPerVolume = insz(3); 
images   = zeros(insz(1),insz(2),nChannelsPerVolume,nFiles, 'uint8');
labels   = zeros(insz(1),insz(2),nChannelsPerVolume,nFiles, 'uint8');
tmpSeg   = zeros(insz(1),insz(2),nChannelsPerVolume, 'uint8');
% ibsrLabels = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,29,30,41,...
%     42,43,44,46,47,48,49,50,51,52,53,54,58,60,61,62,72];
ibsrLabels = [0, 10, 11, 12, 13, 49, 50, 51, 52];
labelMap = containers.Map(ibsrLabels,0:numel(ibsrLabels)-1);

if exist('data/ibsr.mat', 'file') ~= 2,
    ticStart = tic;
    for i=1:nFiles
        imgPath = fullfile(paths.IBSR, dirs(i).name, [dirs(i).name '_ana_strip.nii']);
        segPath = fullfile(paths.IBSR, dirs(i).name, [dirs(i).name '_seg_ana.nii']);
        if ~exist(imgPath,'file') && exist([imgPath '.gz'],'file')
            gunzip([imgPath '.gz']);
        end
        if ~exist(segPath,'file') && exist([segPath '.gz'],'file')
            gunzip([segPath '.gz']);
        end
        img = load_nii(imgPath);
        seg = load_nii(segPath);
        img.img = permute(img.img, permvec);
        seg.img = permute(seg.img, permvec);
        assert(size(img.img,3) == nChannelsPerVolume)
        tmpSeg(:) = 0;
        for j=1:numel(ibsrLabels)
            tmpSeg(seg.img == ibsrLabels(j)) = labelMap(ibsrLabels(j));
        end
        images(:,:,:,i) = 255*bsxfun(@rdivide,single(img.img), single(max(max(img.img))));
        labels(:,:,:,i) = tmpSeg;
        progress('Reading ISBR images ',i,nFiles,ticStart);
    end
    save('data/ibsr.mat', 'images', 'labels');
else
    load('data/ibsr.mat', 'images', 'labels');
    fprintf('Loaded cached ISBR images from data/ibsr.mat\n')
end

% VERSION OF THE CODE WORKING ON MHD images -------------------------------
% nFiles   = 18; 
% insz     = [158,123,145]; 
% images   = zeros(insz(1),insz(2),nChannelsPerVolume,nFiles, 'uint8');
% labels   = zeros(insz(1),insz(2),nChannelsPerVolume,nFiles, 'uint8');
% tmpSeg   = zeros(insz(1),insz(2),nChannelsPerVolume, 'uint8');
% imgFiles = dir(fullfile(paths.IBSR.images,'*.mhd'));
% segFiles = dir(fullfile(paths.IBSR.labels,'*.mhd'));
% assert(numel(imgFiles) == numel(segFiles))
% ibsrLabels = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,29,30,41,...
%     42,43,44,46,47,48,49,50,51,52,53,54,58,60,61,62,72];
% labelMap = containers.Map(ibsrLabels,0:numel(ibsrLabels)-1);
% ticStart = tic;
% for i=1:nFiles
%     img = read_mhd(fullfile(paths.IBSR.images, imgFiles(i).name));
%     seg = read_mhd(fullfile(paths.IBSR.labels, segFiles(i).name));
%     img.data = permute(img.data, permvec);
%     seg.data = permute(seg.data, permvec);
%     assert(size(img.data,3) == nChannelsPerVolume)
%     tmpSeg(:) = 0;
%     for j=1:numel(ibsrLabels)
%         tmpSeg(seg.data == ibsrLabels(j)) = labelMap(ibsrLabels(j));
%     end
%     images(:,:,:,i) = 255*bsxfun(@rdivide,img.data,max(max(img.data)));
%     labels(:,:,:,i) = tmpSeg;
%     progress('Reading ISBR images',i,nFiles,ticStart);
% end

if cropped
    [imin, imax, jmin, jmax, kmin, kmax] = findbbox(images, 3);
    images = images(imin:imax, jmin:jmax, kmin:kmax, :);
    labels = labels(imin:imax, jmin:jmax, kmin:kmax, :);
    insz = size(images);
    insz = insz(1:3);
    nChannelsPerVolume = insz(3);
end

imagesTrain = reshape(images(:,:,:,indsTrain),insz(1),insz(2),[]);
imagesVal   = reshape(images(:,:,:,indsVal),  insz(1),insz(2),[]); clear images;
labelsTrain = reshape(labels(:,:,:,indsTrain),insz(1),insz(2),[]);
labelsVal   = reshape(labels(:,:,:,indsVal),  insz(1),insz(2),[]); clear labels;
assert(size(imagesTrain,3) == numel(indsTrain) * nChannelsPerVolume);
assert(size(imagesVal  ,3) == numel(indsVal) * nChannelsPerVolume);


% Augment train patches
% Flip horizontally
imagesTrain = cat(3, imagesTrain, flipdim(imagesTrain, 2));
labelsTrain = cat(3, labelsTrain, flipdim(labelsTrain, 2));
if augment
    imagesTrainAug = imagesTrain;
    labelsTrainAug = labelsTrain;
    % Shift images
    for shift = [5 10 15 20]
        imagesTrainAug = cat(3, shiftim(imagesTrain, [shift 0]), shiftim(imagesTrain, [-shift 0]), ...
                        shiftim(imagesTrain, [0 shift]),      shiftim(imagesTrain, [0 -shift]),...
                        shiftim(imagesTrain, [shift shift]),  shiftim(imagesTrain, [-shift -shift]),...
                        shiftim(imagesTrain, [shift -shift]), shiftim(imagesTrain, [-shift shift]), imagesTrainAug);
        labelsTrainAug = cat(3, shiftim(labelsTrain, [shift 0]), shiftim(labelsTrain, [-shift 0]), ...
                        shiftim(labelsTrain, [0 shift]),      shiftim(labelsTrain, [0 -shift]),...
                        shiftim(labelsTrain, [shift shift]),  shiftim(labelsTrain, [-shift -shift]),...
                        shiftim(labelsTrain, [shift -shift]), shiftim(labelsTrain, [-shift shift]), labelsTrainAug);
    end
    imagesTrain = imagesTrainAug; clear imagesTrainAug;
    labelsTrain = labelsTrainAug; clear labelsTrainAug;
end
insz        = net.meta.normalization.imageSize(1:2);       % e.g. 321x321
outsz       = net.meta.outputSize;                         % e.g. 41x41
labelsTrain = imresize(labelsTrain, outsz, 'nearest');
labelsVal   = imresize(labelsVal,   outsz, 'nearest');
if ~isequal(insz,[size(imagesTrain,1),size(imagesTrain,2)])
    imagesTrain = imresize(imagesTrain, insz, 'bicubic');
    imagesVal   = imresize(imagesVal,   insz, 'bicubic');
end

nSlicesTrain= size(imagesTrain,3);
nSlicesVal  = size(imagesVal,3);
if net.meta.normalization.imageSize(3) == 1  % single channel per training example
    imdb.train = 1:nSlicesTrain;
    imdb.val   = (1:nSlicesVal) + nSlicesTrain;
elseif net.meta.normalization.imageSize(3) > 1   % multiple channels per training example
    % We will store the indices of the slides corresponding to each stack
    % to avoid replicating training data and keep RAM requirements low.
    assert(isodd(net.meta.normalization.imageSize(3)),'The number of input channels must be odd');
    assert(~mod(nSlicesTrain,nChannelsPerVolume))
    assert(~mod(nSlicesVal,  nChannelsPerVolume))
    nVolumesTrain = nSlicesTrain/nChannelsPerVolume;
    nVolumesVal = nSlicesVal/nChannelsPerVolume;
    stackWidth  = (net.meta.normalization.imageSize(3) - 1)/2;
    imdb.train  = bsxfun(@plus, 1:nChannelsPerVolume, (-stackWidth:stackWidth)');
    imdb.train  = imdb.train(:,stackWidth+1:end-stackWidth);
    imdb.val    = imdb.train;
    assert(isinrange(imdb.train,[1, nChannelsPerVolume]), 'Indexes out of range')
    nExamplesPerVolume = size(imdb.train,2);
    % Expand indices for the rest of training examples
    imdb.train  = repmat(imdb.train, [1 nVolumesTrain]);
    imdb.val    = repmat(imdb.val,   [1 nVolumesVal]);
    for i=0:(nVolumesTrain-1)
        imdb.train(:,(1:nExamplesPerVolume)+i*nExamplesPerVolume) = ...
            imdb.train(:,(1:nExamplesPerVolume)+i*nExamplesPerVolume) + nChannelsPerVolume*i;
    end
    for i=0:(nVolumesVal-1)
        imdb.val(:,(1:nExamplesPerVolume)+i*nExamplesPerVolume) = ...
            imdb.val(:,(1:nExamplesPerVolume)+i*nExamplesPerVolume) + nChannelsPerVolume*i;
    end
    imdb.val = imdb.val + nSlicesTrain;
end
% This reshape is necessary for vl_softmaxloss to work properly.
labelsTrain = reshape(labelsTrain, outsz(1),outsz(2),1,[]);
labelsVal   = reshape(labelsVal,   outsz(1),outsz(2),1,[]);
imagesTrain = reshape(imagesTrain, insz(1), insz(2), 1,[]);
imagesVal   = reshape(imagesVal,   insz(1), insz(2), 1,[]);
imagesTrain = cat(4, imagesTrain, imagesVal); clear imagesVal;
labelsTrain = cat(4, labelsTrain, labelsVal); clear labelsVal;

imdb.images = imagesTrain; 
imdb.labels = labelsTrain+1; % add 1 for vl_nnsoftmaxloss to work
assert(isinrange(imdb.labels,[1,numel(ibsrLabels)]),'Labels not in range')
assert(size(imdb.images,4) == size(imdb.labels,4))
assert(max(imdb.val(:)) == size(imdb.images,4))

save('data/imdb.mat', 'imdb');
fprintf('Saved imdb to data/imdb.mat\n')
end