function cnn(dataset,indsTrain,indsVal,tag,view,varargin)
% CNN Setup and train a CNN on IBSR or RE dataset. Based on the MatConvNet
% template.
% 
% CNN(dataset,indsTrain, indsVal) where dataset is either 'IBSR' (Internet
% Brain Segmentation Repository) or 'RE' (Roland Epilepsy) trains a CNN
% using data from the respective dataset. IndsTrain are the indices of the
% subjects used for trianing and indsVal are the indices of the subjects
% used for validation.
% 
% CNN(dataset,indsTrain, indsVal, tag) adds a tag in the name of the stored
% model file.
% 
% CNN(dataset,indsTrain, indsVal, tag, view) uses slices from one out of
% three possible views: 'axial' (1), 'sagittal' (2), or 'coronal' (3).
% 
% CNN(dataset,indsTrain, indsVal, tag, view, varargin) uses additional
% arguments for the MatConvNet functions.
% 
% Stavros Tsogkas, <stavros.tsogkas@centralesupelec.fr>
% Last update: February 2016

% Default arguments
if nargin < 1, dataset    = 'IBSR'; end
if nargin < 2, indsTrain  = 1:10;   end
if nargin < 3, indsVal    = 11:12;  end
if nargin < 4, tag        = '';     end
if nargin < 5, view       = 1;      end

% Default options for CNN training
opts.modelType = 'dropout' ; % bnorm or dropout
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.expDir = 'results/cnn';
opts.subtractMean = 0;  % calculate and subtract mean image from each training image
opts.train.batchSize = 10; % increase this value if you have enough GPU RAM
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = [1];
opts.train.prefetch = true ;
opts.train.sync = true ;
opts.train.expDir = opts.expDir ;
opts.train.plotErrors = 1;
opts.train.modelName = ['net-' dataset tag];
switch opts.modelType
  case 'dropout', opts.train.learningRate = logspace(-2, -4, 50) ;
  case 'bnorm',   opts.train.learningRate = logspace(-1, -3, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

% Network and  Database initialization
disp('Database initialization')
switch dataset
    case 'IBSR'
        net  = cnnIBSRv2Init();
        imdb = setupImdbIBSRv2(net,indsTrain,indsVal,0,view);
    case 'RE'
        warning('RE is a proprietary dataset. Any RE-related code is included for archiving reasons.')
        net = cnnREInit();
        rng(0); nFiles = 35; inds = randperm(nFiles);
        if indsTrain == 1
            indsTrain = inds(1:18); indsVal = inds(19:20);
        elseif indsTrain == 2
            indsTrain = inds(19:end); indsVal = inds(1:2);
        end
        imdb = setupImdbRE(net,indsTrain,indsVal);
end
if opts.subtractMean
    net.meta.normalization.averageImage = mean(imdb.images,4);
    imdb.images = bsxfun(@minus, imdb.images, net.meta.normalization.averageImage);
end

bopts = net.meta.normalization ;
bopts.numThreads = opts.numFetchThreads ;
opts.train.backPropDepth = net.meta.backPropDepth;

% Stochastic gradient descent
bopts.transformation = 'stretch' ;
fn = getBatchWrapper(bopts) ;

[net,info] = cnnTrain(net, imdb, fn, opts.train, 'conserveMemory', true) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,segs] = getBatch(imdb, batch, opts)
% -------------------------------------------------------------------------
im   = single(imdb.images(:,:,:,batch));
segs = single(imdb.labels(:,:,:,batch));

% -------------------------------------------------------------------------
function [net, info] = cnnTrain(net, imdb, getBatch, varargin)
% -------------------------------------------------------------------------
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.gpus = [] ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {'top1e'} ;
opts.plotErrors = true;
opts.plotDiagnostics = false ;
opts.modelName = 'net';
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = imdb.train; end
if isempty(opts.val), opts.val = imdb.val; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
  end
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error calculation function
if ischar(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
    otherwise
      error('Uknown error function ''%s''', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

for epoch=1:opts.numEpochs
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to last checkpoint
  modelPath = @(ep) fullfile(opts.expDir, sprintf('%s-epoch-%d.mat',opts.modelName, ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        load(modelPath(epoch), 'net', 'info') ;
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end

  % move CNN to GPU as needed
  if numGpus == 1
    net = vl_simplenn_move(net, 'gpu') ;
  elseif numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net, 'gpu') ;
    end
  end

  % train one epoch and validate
  % opts.train and opts.val must be either matrices or row vectors
  assert(size(opts.train,2) > 1 && size(opts.val,2) > 1) 
  train = opts.train(:,randperm(size(opts.train,2))) ; % shuffle
  val   = opts.val ;
  if numGpus <= 1
    [net,stats.train] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net) ;
    [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net) ;
  else
    spmd(numGpus)
      [net_, stats_train_] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net_) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_) ;
    end
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
  end

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
%     n = numel(eval(f)) ;
    n = size(eval(f),2) ;   % TSOGKAS: change to make it work with multi-slice data
    info.(f).speed(epoch) = n / stats.(f)(1) ;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
  end
  if numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net_, 'cpu') ;
    end
    net = net_{1} ;
  else
    net = vl_simplenn_move(net, 'cpu') ;
  end
  if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end

  if opts.plotErrors
      figure(1) ; clf ;
      hasError = isa(opts.errorFunction, 'function_handle') ;
      subplot(1,1+hasError,1) ;
      if ~evaluateMode
          semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
          hold on ;
      end
      semilogy(1:epoch, info.val.objective, '.--') ;
      xlabel('training epoch') ; ylabel('energy') ;
      grid on ;
      h=legend(sets) ;
      set(h,'color','none');
      title('objective') ;
      if hasError
          subplot(1,2,2) ; leg = {} ;
          if ~evaluateMode
              plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
              hold on ;
              leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
          end
          plot(1:epoch, info.val.error', '.--') ;
          leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
          set(legend(leg{:}),'color','none') ;
          grid on ;
          xlabel('training epoch') ; ylabel('error') ;
          title('error') ;
      end
      drawnow ;
      print(1, modelFigPath, '-dpdf') ;
  end
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = max(predictions,[],3); 
err = nnz(predictions ~= labels);

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net,stats,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net)
% -------------------------------------------------------------------------

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else mode = 'validation' ; end
if training, simplemode = 'normal' ; else simplemode = 'test' ; end
if nargout > 2, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = [] ;

numelSubset = size(subset,2);
for t=1:opts.batchSize:numelSubset
  fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numelSubset/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numelSubset - t + 1) ;
  batchTime = tic ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numelSubset) ;
    batch = subset(:,batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    [im, labels] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numelSubset) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(:,batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    % evaluate CNN
    net.layers{end}.class = labels ;
    if training, dzdy = one; else dzdy = [] ; end
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'mode', simplemode, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync) ;
                  
    %  Testing gpu/cpu results ------------------------------------------------
    if 0
      resgpu = vl_simplenn(net, im, dzdy, res, ...
          'accumulate', s ~= 1, ...
          'mode', simplemode, ...
          'conserveMemory', opts.conserveMemory, ...
          'backPropDepth', opts.backPropDepth, ...
          'sync', opts.sync) ;
      net.layers{end} = rmfield(net.layers{end},'class'); net = vl_simplenn_move(net,'cpu'); net.layers{end}.class = labels;
      rescpu = vl_simplenn(net, gather(im), gather(dzdy), res, ...
          'accumulate', s ~= 1, ...
          'mode', simplemode, ...
          'conserveMemory', opts.conserveMemory, ...
          'backPropDepth', opts.backPropDepth, ...
          'sync', opts.sync) ;
      assert(numel(resgpu) == numel(rescpu));
      tol = 1e-2;
      for i=numel(resgpu):-1:1
          if ~isempty(resgpu(i).x) && ~isempty(rescpu(i).x)
              assert(isequaltol(resgpu(i).x,rescpu(i).x,tol))
          end
          if ~isempty(resgpu(i).dzdx) && ~isempty(rescpu(i).dzdx)
              assert(isequaltol(resgpu(i).dzdx,rescpu(i).dzdx,tol))
          end
          if ~isempty(resgpu(i).dzdw) && ~isempty(rescpu(i).dzdw)
              assert(isequaltol(resgpu(i).dzdw{1},rescpu(i).dzdw{1},tol))
              assert(isequaltol(resgpu(i).dzdw{2},rescpu(i).dzdw{2},tol))
          end
      end
      res = resgpu;
      net.layers{end} = rmfield(net.layers{end},'class'); net = vl_simplenn_move(net,'gpu'); net.layers{end}.class = labels;
    end

    % accumulate training errors
    error = sum([error, [...
      sum(double(gather(res(end).x))) ;
      reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
    numDone = numDone + numel(batch) ;
  end

  % gather and accumulate gradients across labs
  if training
    if numGpus <= 1
      net = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
    end
  end

  % print learning statistics
  batchTime = toc(batchTime) ;
  stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
  speed = batchSize/batchTime ;

  fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
  n = (t + batchSize - 1) / max(1,numlabs) ;
  fprintf(' obj:%.3g', stats(2)/n) ; if isnan(stats(2)/n), keyboard; end
  for i=1:numel(opts.errorLabels)
    fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
  end
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
end

if nargout > 2
  prof = mpiprofile('info');
  mpiprofile off ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=1:numel(net.layers)
  for j=1:numel(res(l).dzdw)
    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
    thisLR = lr * net.layers{l}.learningRate(j) ;

    % accumulate from multiple labs (GPUs) if needed
    if nargin >= 6
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end

    if isfield(net.layers{l}, 'weights')
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
    end
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end
