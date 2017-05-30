function shifted = shiftim(images, shifts)
% only with 2d shifts
if length(shifts) ~= 2
    error('Not implemented')
end
nd = ndims(images);
index = repmat({':'},1,nd-2);

shifted = circshift(images, shifts);

if shifts(1) >= 0
    shifted(1:shifts(1), :, index{:}) = 0;
else
    shifted(end + shifts(1) + 1:end, :, index{:}) = 0;
end
if shifts(2) >= 0
    shifted(:, 1:shifts(2), index{:}) = 0;
else
    shifted(:, end + shifts(2) + 1:end, index{:}) = 0;
end

end