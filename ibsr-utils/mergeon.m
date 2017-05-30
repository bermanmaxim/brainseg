function image = mergeon(gt, segmentation)
seg_r = segmentation(:, :, 1);
seg_g = segmentation(:, :, 2);
seg_b = segmentation(:, :, 3);

allnul = seg_r == 0 & seg_g == 0 & seg_b == 0;

image_r = gt;
image_g = gt;
image_b = gt;

image_r(~allnul) = seg_r(~allnul);
image_g(~allnul) = seg_g(~allnul);
image_b(~allnul) = seg_b(~allnul);

image = cat(3, image_r, image_g, image_b);
end