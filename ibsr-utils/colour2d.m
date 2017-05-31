function image = colour2d(segmentation, restrict_to, invlabelMap)

if nargin < 2
    restrict_to = [10, 11, 12, 13, 49, 50, 51, 52];
end

if nargin < 3
    invlabelMap = [];
end


labelprops = getlabels;

image_r = zeros(size(segmentation, 1), size(segmentation, 2), 'uint8');
image_g = zeros(size(segmentation, 1), size(segmentation, 2), 'uint8');
image_b = zeros(size(segmentation, 1), size(segmentation, 2), 'uint8');

unilab = unique(segmentation);
for j = 1:length(unilab)
    i = unilab(j);
    if ~isempty(restrict_to) && ~ismember(i, restrict_to)
        continue
    end
    if isKey(invlabelMap, i)
        k = invlabelMap(i);
    else
        k = i;
    end
    image_r(segmentation == i) = labelprops(k).rgb(1);
    image_g(segmentation == i) = labelprops(k).rgb(2);
    image_b(segmentation == i) = labelprops(k).rgb(3);
end

image = cat(3, image_r, image_g, image_b);

end
