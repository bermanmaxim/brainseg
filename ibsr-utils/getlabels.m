function labelMap = getlabels

labels = {
  { 0, 'Unknown',                                  0,    0,    0,}
  { 2, 'Left-Cerebral-White-Matter',             225,  225,  225,}
  { 3, 'Left-Cerebral-Cortex',                   205,   62,   78,}
  { 4, 'Left-Lateral-Ventricle',                 120,   18,  134,}
  { 5, 'Left-Inf-Lat-Vent',                      196,   58,  250,}
  { 7, 'Left-Cerebellum-White-Matter',           220,  248,  164,}
  { 8, 'Left-Cerebellum-Cortex',                 230,  148,   34,}
  {10, 'Left-Thalamus-Proper',                     0,  118,   14,}
  {11, 'Left-Caudate',                           122,  186,  220,}
  {12, 'Left-Putamen',                           236,   13,  176,}
  {13, 'Left-Pallidum',                           12,   48,  255,}
  {14, '3rd-Ventricle',                          204,  182,  142,}
  {15, '4th-Ventricle',                           42,  204,  164,}
  {16, 'Brain-Stem',                             119,  159,  176,}
  {17, 'Left-Hippocampus',                       220,  216,   20,}
  {18, 'Left-Amygdala',                          103,  255,  255,}
  {24, 'CSF',                                     60,   60,   60,}
  {26, 'Left-Accumbens-area',                    255,  165,    0,}
  {28, 'Left-VentralDC',                         165,   42,   42,}
  {29, 'Left-undetermined',                      135,  206,  235,}
  {30, 'Left-vessel',                            160,   32,  240,}
  {41, 'Right-Cerebral-White-Matter',              0,  225,    0,}
  {42, 'Right-Cerebral-Cortex',                  205,   62,   78,}
  {43, 'Right-Lateral-Ventricle',                120,   18,  134,}
  {44, 'Right-Inf-Lat-Vent',                     196,   58,  250,}
  {46, 'Right-Cerebellum-White-Matter',          220,  248,  164,}
  {47, 'Right-Cerebellum-Cortex',                230,  148,   34,}
  {48, 'Right-Thalamus',                           0,  118,   14,}
  {49, 'Right-Thalamus-Proper',                    0,  118,   14,}
  {50, 'Right-Caudate',                          122,  186,  220,}
  {51, 'Right-Putamen',                          236,   13,  176,}
  {52, 'Right-Pallidum',                         255,   48,  255,}
  {53, 'Right-Hippocampus',                      220,  216,   20,}
  {54, 'Right-Amygdala',                         103,  255,  255,}
  {58, 'Right-Accumbens-area',                   255,  165,    0,}
  {60, 'Right-VentralDC',                        165,   42,   42,}
  {61, 'Right-undetermined',                     135,  206,  235,}
  {62, 'Right-vessel',                           160,   32,  240,}
  {72, '5th-Ventricle',                          120,  190,  150,}
};

labelMap = containers.Map('KeyType', 'int64', 'ValueType', 'any');
for i = 1:length(labels)
    line = labels{i};
    ind.name = line{2};
    ind.rgb = cell2mat(line(3:end));
    labelMap(line{1}) = ind;
end

end