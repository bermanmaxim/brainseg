function image = colour2d(segmentation, restrict_to)
% using original isbr labels!

if nargin < 2
    restrict_to = [10, 11, 12, 13, 49, 50, 51, 52];
end

labelMap = getlabels;

image_r = zeros(size(segmentation, 1), size(segmentation, 2), 'uint8');
image_g = zeros(size(segmentation, 1), size(segmentation, 2), 'uint8');
image_b = zeros(size(segmentation, 1), size(segmentation, 2), 'uint8');

unilab = unique(segmentation);
for j = 1:length(unilab)
    i = unilab(j);
    if ~isempty(restrict_to) && ~ismember(i, restrict_to)
        continue
    end
    image_r(segmentation == i) = labelMap(i).rgb(1);
    image_g(segmentation == i) = labelMap(i).rgb(2);
    image_b(segmentation == i) = labelMap(i).rgb(3);
end

image = cat(3, image_r, image_g, image_b);

end
