function errors = evalTest

load('data/ibsr.mat', 'images', 'labels');

indsTrain  = 1:10;
indsVal    = 11:12;
indsTest  = setdiff(1:18, union(indsTrain, indsVal));
numGpus = 1;
opts.continue = true;
opts.expDir = 'results/cnn';

imagesTest = images(:, :, :, indsTest);
labelsTest = labels(:, :, :, indsTest);

net  = cnnIBSRv2Init();
if numGpus >= 1
  net = vl_simplenn_move(net, 'gpu') ;
end

insz        = net.meta.normalization.imageSize(1:2);       % 256x256
% outsz       = net.meta.outputSize;                         % 64x64

imagesTest = imresize(imagesTest, insz, 'nearest');

% labelsTrain = imresize(labelsTrain, outsz, 'nearest');
% labelsVal   = imresize(labelsVal,   outsz, 'nearest');
% if ~isequal(insz,[size(imagesTrain,1),size(imagesTrain,2)])
%     imagesTrain = imresize(imagesTrain, insz, 'bicubic');
%     imagesVal   = imresize(imagesVal,   insz, 'bicubic');
% end

im = imagesTest(:, :, :, 1);

if numGpus >= 1
  im = gpuArray(im) ;
end


modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: loading last saved epoch %d\n', mfilename, start) ;
  net = loadState(modelPath(start)) ;
else
  state = [] ;
end
im = reshape(im, insz(1), insz(2), [], size(im, 3));

res = vl_simplenn(net, im) ;

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
