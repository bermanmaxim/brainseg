slice = 128;
ibsrLabels = [0, 10, 11, 12, 13, 49, 50, 51, 52];

labelMap = containers.Map(ibsrLabels,0:numel(ibsrLabels)-1);
invlabelMap = containers.Map(0:numel(ibsrLabels)-1, ibsrLabels);

i = 1;
for slice = 50:200
    labl = zeros(size(labels, 1), size(labels, 2), 'int64');
    for j=0:numel(ibsrLabels)-1
        labl(labels(:, :, slice, i) == j) = invlabelMap(j);
    end
    coloured = mergeon(images(:, :, slice, i), colour2d(labl));
    imshow(coloured)
    pause
end

