slice = 128;

ibsrLabels = [0,2,3,4,5,7,8,10,11,12,13,14,15,16,17,18,24,26,28,29,30,41,...
    42,43,44,46,47,48,49,50,51,52,53,54,58,60,61,62,72];

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

