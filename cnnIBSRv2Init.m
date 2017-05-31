function net = cnnIBSRv2Init(varargin)
% CNN_IMAGENET_INIT  Baseline CNN model

opts.scale = 1 ;
opts.initBias = 0.1 ;
opts.weightDecay = 1 ;
opts.labelset = 'set9';
opts = vl_argparse(opts, varargin) ;
% opts.labelindices = [0, 10, 11, 12, 13, 49, 50, 51, 52]; % ibsr labels

if strcmp(opts.labelset, 'set9')
    opts.labelindices = [0, 10, 11, 12, 13, 49, 50, 51, 52]; % ibsr labels
elseif strcmp(opts.labelset, 'set39')
    opts.labelindices = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,29,30,41,...
    42,43,44,46,47,48,49,50,51,52,53,54,58,60,61,62,72];
end

lmap = getlabels;
opts.labelnames = {};
for i = opts.labelindices
    opts.labelnames{end + 1} = lmap(i).name;
end

opts.nLabels = length(opts.labelindices) ;

net.layers = {} ;

% Define input and output size
net.normalization.imageSize = [256, 256, 1] ;
net.normalization.interpolation = 'bilinear' ;
net.normalization.averageImage = [] ;
net.normalization.keepAspect = true ;
net.nLabels = opts.nLabels;

% Block 1
net = addConvBlock(net, opts, 1, 7, 7, net.normalization.imageSize(3), 64, 1, 3, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool1', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;

% Block 2
net = addConvBlock(net, opts, 2, 5, 5, 64, 128, 1, 2, 1) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool2', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 1) ;

% Block 3
net = addConvBlock(net, opts, 3, 3, 3, 128, 256, 1, 2, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool3', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout3', 'rate', 0.5) ;

% Block 4
net = addConvBlock(net, opts, 4, 3, 3, 256, 512, 1, 2, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool4', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout4', 'rate', 0.5) ;

% Block 5
net = addConvBlock(net, opts, 5, 3, 3, 512, 512, 1, 2, 2) ;
net.layers{end+1} = struct('type', 'pool', 'name', 'pool5', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 1, ...
                           'pad', 1) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout5', 'rate', 0.5) ;

% Block 6
net = addConvBlock(net, opts, 6, 4, 4, 512, 1024, 1, 6, 4) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'dropout6', 'rate', 0.5) ;


% Block 7
net = addConvBlock(net, opts, 7, 1, 1, 1024, opts.nLabels, 1, 0, 1) ;
net.layers(end) = [] ;  % remove last relu layer

% Block 8
% net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
[forw, backw] = segmentationloss;
net.layers{end+1} = struct('type', 'custom', 'forward', forw, 'backward', backw,...
                           'weights', []) ;
net = vl_simplenn_tidy(net) ; % upgrade to last version
net.meta.outputSize = [64 64]; % height and width of the final convolutional layer
net.meta.backPropDepth = inf;
net.meta.labelindices = opts.labelindices;
net.meta.labelnames = opts.labelnames;
net.meta.labelset = opts.labelset;
net.layers{end-1}.precious = true; % keep this intermediate result


function net = addConvBlock(net, opts, id, h, w, in, out, stride, pad, hole)
if nargin < 10, hole = 1; end
if nargin < 9, pad = 0; end
if nargin < 8, stride = 1; end
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
if ischar(id)
    convName = sprintf('%s%s', name, id);
    reluName = sprintf('relu%s',id);
else
    convName = sprintf('%s%d', name, id);
    reluName = sprintf('relu%d',id);
end
net.layers{end+1} = struct('type', 'conv', 'name', convName, ...
                           'weights', {{0.01/opts.scale * randn(h, w, in, out, 'single'), zeros(1, out,'single')}}, ...
                           'stride', stride, ...
                           'pad', pad, ...
                           'dilate', hole,...
                           'learningRate', [1 2], ...
                           'weightDecay', [opts.weightDecay 0]) ;
net.layers{end+1} = struct('type', 'relu', 'name', reluName) ;
