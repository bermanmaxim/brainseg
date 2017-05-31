function [ forw, backw ] = segmentationloss

forw = @forward;
backw = @backward;

end

function resout = forward(layer, resin, resout)
  mass = sum(sum(layer.class > 0,2),1) + 1 ;
  resout.x = vl_nnloss(resin.x, layer.class, [], ...
                       'loss', 'softmaxlog', ...
                       'instanceWeights', 1./mass) ;
end

function resin = backward(layer, resin, resout)
  mass = sum(sum(layer.class > 0,2),1) + 1 ;
  resin.dzdx = vl_nnloss(resin.x, layer.class, resout.dzdx, ...
                         'loss', 'softmaxlog', ...
                         'instanceWeights', 1./mass) ;
end
