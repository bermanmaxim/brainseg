function merged = mergeon(image, segmentation)
% if nargin < 3, resize = 2; end
% merge segmentation on ground truth image
% in case of different sizes, resize to common size

if size(image, 1) ~= size(segmentation, 1) || size(image, 2) ~= size(segmentation, 2)
    imsize = size(image);
    image = imresize(image, [size(segmentation, 1) size(segmentation, 2)]);
    resized = true;
end

seg_r = segmentation(:, :, 1);
seg_g = segmentation(:, :, 2);
seg_b = segmentation(:, :, 3);

allnul = seg_r == 0 & seg_g == 0 & seg_b == 0;

image_r = image;
image_g = image;
image_b = image;

image_r(~allnul) = seg_r(~allnul);
image_g(~allnul) = seg_g(~allnul);
image_b(~allnul) = seg_b(~allnul);

merged = cat(3, image_r, image_g, image_b);

if resized
    merged = imresize(merged, imsize);
end

end