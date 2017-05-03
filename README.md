# FunctionalDataUtils

[![Build Status](https://travis-ci.org/rened/FunctionalDataUtils.jl.png)](https://travis-ci.org/rened/FunctionalDataUtils.jl)
[![Build Status](http://pkg.julialang.org/badges/FunctionalDataUtils_0.4.svg)](http://pkg.julialang.org/?pkg=FunctionalDataUtils&ver=0.4)
[![Build Status](http://pkg.julialang.org/badges/FunctionalDataUtils_0.5.svg)](http://pkg.julialang.org/?pkg=FunctionalDataUtils&ver=0.5)
[![Build Status](http://pkg.julialang.org/badges/FunctionalDataUtils_0.6.svg)](http://pkg.julialang.org/?pkg=FunctionalDataUtils&ver=0.6)

Utility functions based on [FunctionData.jl](http://github.com/rened/FunctionalData.jl), mostly from the area of computer vision and machine learning.

#### Numerical
```jl
normsum, normsum!                   # normalize sum to 1
norm01, norm01!                     # normalize to the range 0..1
normeuclid, normeuclid!             # normalize to L2 norm == 1
normmean, normmean!                 # normalize to mean == 0
normmeanstd, normmeanstd!           # normalize to mean == 0, std == 1
normunique(a)                       # replace items with indices to unique(a)
valuemap(a, mapping)                # look up  non-NaNs of a in mapping
pcawhitening                        # perform PCA whitening
zcawhitening                        # perform ZCA whitening
clamp(a, mi, ma)                    # clamp every item to min mi and max ma
nanfunction(f,a,d)                  # apply function f along dim d to non-nan elements of a
nanmean(a,d)                        # mean ignoring NaNs
nanstd(a,d)                         # std ignoring NaNs
nanmedian(a,d)                      # median ignoring NaNs
distance(a[,b])                     # L2 norm between all items in a and b
```

#### Computer Vision

```jl
iimg(a)                             # integral image / volume
iimg!(a)                            # integral image / volume
interp3(a,m,n,o)                    # interpolate a at m,n,o
interp3with01coords(a,m,n,o)        # interpolate a using 0..1 coords
resize(a,siz)                       # resize a 
resizeminmax(a, mi, ma)             # resize a to fit within mi and ma sizes
grid
meshgrid
meshgrid3
centeredgrid
centeredmeshgrid
overlaygradient
toranges
tosize
tosize3
imregionalmin
imregionalmax
monogen
bwlabel
bwlabel!
monoslic
border
bwdist                                      # 
rle(a)                                      # run length encoding
unrle(a)                                    # run lendth decoding
reshape
stridedblockcoords(a, blocksiz, stide)      # return tuples of ranges for each subblock
inpolygon(point, polygon)                   # is point inside polygon? 
inpointcloud(point, cloud)                  # is point inside pointcloud?
```

#### Computing

```jl
@timedone
fasthash
cache
dictcache
loadedmodules
reloadmodules
```

#### Graphics

```jl
jetcolormap(n)                      # 3 x n jet colormap like in Matlab
asimagesc(a)                        # m x n x 3 RGB array with image like Matlab's imagesc
blocksvisu(a)                       # visualization of patches / blocks
```


