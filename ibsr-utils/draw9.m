function draw9(lbl, net, im)
if nargin < 3
    im = [];
end
remainlabs = [0, 10, 11, 12, 13, 49, 50, 51, 52];
ibsrLabels = net.meta.labelindices;
invlabelMap = containers.Map(1:numel(ibsrLabels), ibsrLabels);

insz        = net.meta.normalization.imageSize(1:2);       % 256x256
% outsz       = net.meta.outputSize;                         % 64x64

image = colour2d(lbl, remainlabs, invlabelMap);
if ~isempty(im)
    image = mergeon(uint8(gather(im)), image);
end
imshow(image)
end