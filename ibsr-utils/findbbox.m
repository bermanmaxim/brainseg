function [imin, imax, jmin, jmax, kmin, kmax] = findbbox(images, targetdims)
nonz = abs(images) > 0;

while ndims(nonz) > targetdims
    nonz = sum(nonz, ndims(nonz));
end

nonz = nonz > 0;
if targetdims == 1
    nonzi = find(nonz);
elseif targetdims == 2
    [nonzi, nonzj] = find(nonz);
elseif targetdims == 3
    [nonzi, nonzj, nonzk] = ind2sub(size(nonz),find(nonz));
else
    error('Not Implemented')
end

imin = min(nonzi);
imax = max(nonzi);
if targetdims > 1
    jmin = min(nonzj);
    jmax = max(nonzj);
if targetdims > 2
    kmin = min(nonzk);
    kmax = max(nonzk);
end

end