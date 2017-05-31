function errors = test_cnn

opts.labelset = 'set39';
opts.expName = 'cnn39';

remainlabs = [0, 10, 11, 12, 13, 49, 50, 51, 52];
%remainlabs = [];

load(sprintf('data/ibsr-%s.mat', opts.labelset), 'images', 'labels');

indsTrain  = 1:10;
indsVal    = 11:12;
indsTest  = setdiff(1:18, union(indsTrain, indsVal));
numGpus = 1;
opts.continue = true;
opts.expDir = fullfile('results', opts.expName);

cropped = true;

if cropped
    [imin, imax, jmin, jmax, kmin, kmax] = findbbox(images, 3);
    images = images(:, :, kmin:kmax, :);
    labels = labels(:, :, kmin:kmax, :);
end

imagesTest = images(:, :, :, indsTest);
labelsTest = labels(:, :, :, indsTest);

net  = cnnIBSRv2Init('labelset', opts.labelset);

ibsrLabels = net.meta.labelindices;
ibsrLabelsNames = net.meta.labelnames;
invlabelMap = containers.Map(1:numel(ibsrLabels), ibsrLabels);

insz        = net.meta.normalization.imageSize(1:2);       % 256x256
% outsz       = net.meta.outputSize;                         % 64x64

imagesTest = imresize(imagesTest, insz, 'bicubic');

% labelsTrain = imresize(labelsTrain, outsz, 'nearest');
% labelsVal   = imresize(labelsVal,   outsz, 'nearest');
% if ~isequal(insz,[size(imagesTrain,1),size(imagesTrain,2)])
%     imagesTrain = imresize(imagesTrain, insz, 'bicubic');
%     imagesVal   = imresize(imagesVal,   insz, 'bicubic');
% end

slice = 80;
patient = 1;

im = imagesTest(:, :, slice, patient);
im = single(reshape(im, insz(1), insz(2), 1, 1));
lab = labelsTest(:, :, slice, patient);
lab = reshape(lab, size(lab, 1), size(lab, 2), 1, 1);

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));

start = opts.continue * findLastCheckpoint(opts.expDir) ;

%start = 1;

if start >= 1
  fprintf('%s: loading last saved epoch %d\n', mfilename, start) ;
  net = loadState(modelPath(start)) ;
else
  state = [] ;
end

if numGpus >= 1
  net = vl_simplenn_move(net, 'gpu') ;
end

net.layers(end) = []; % remove softmax loss layer

if numGpus >= 1
  im = gpuArray(im) ;
end

res = vl_simplenn(net, im, [], [], ...
                  'mode', 'test', ...
                  'ConserveMemory', false) ;
out = vl_nnsoftmax(res(end).x) ;
out = imresize(out, [size(lab, 1), size(lab, 2)]) ;

[prob, amax] = max(out, [], 3);

amax = gather(amax);

figure
subplot(2, 1, 1)
title('ground truth')
image = mergeon(uint8(gather(im)), colour2d(lab, remainlabs, invlabelMap));
imshow(image)

intersect(unique(lab), remainlabs)

subplot(2, 1, 2)
title('predicted')
image = mergeon(uint8(gather(im)), colour2d(amax, remainlabs, invlabelMap));
imshow(image)

end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = vl_simplenn_tidy(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end
end
