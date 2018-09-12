export normsum, norm01, normeuclid, normmean, normmeanstd
export normsum!, norm01!, normeuclid!, normmean!, normmeanstd!
export normquantile
export normunique, valuemap, unvaluemap
export normmean_
export pcawhitening, zcawhitening, WhiteningBase
export clamp, clamp!
export nanfunction, nanmean, nanstd, nanmedian
export distance
export randbase, randnormbase, randproj
export mean_, std_, var_, min_, max_, sum_, median_, dropdims_


#######################################
##  normsum, norm01, normeuclid, normmeanstd
##  normsum!, norm01!, normeuclid!, normmeanstd!

copyfloat(a::Array{T}) where {T<:Int} = float(a)
copyfloat(a::Array{T}) where {T} = copy(a)

normsum(a) = normsum!(copyfloat(a))
function normsum!(r)
    a = sum(r)
    a == 0 && return r
    for i = 1:length(r)
        r[i] /= a
    end
    r
end

norm01(a) = norm01!(copyfloat(a))
function norm01!(r)
    mi = minimum(r)
    ma = maximum(r)
    d = ma - mi
    for i = 1:length(r)
        r[i] -= mi
    end
    if d>0
        for i = 1:length(r)
            r[i] /= d
        end
    end
    r
end

normeuclid(a) = normeuclid!(copyfloat(a))
function normeuclid!(r)
    a = zeroel(r)
    for i = 1:length(r)
        a += r[i]*r[i]
    end
    a = sqrt(a)

    if a > 0
        for i = 1:length(r)
            r[i] /= a
        end
    end
    r
end

normmean(a) = normmean!(copyfloat(a))
function normmean!(r)
    a = mean(r)
    for i = 1:length(r)
        r[i] /= a
    end
    r
end

normmeanstd(a) = normmeanstd!(copyfloat(a))
function normmeanstd!(r)
    m = mean(r)
    s = std(r)
    for i = 1:length(r)
        r[i] -= m
    end
    if s > 0
        for i = 1:length(r)
            r[i] /= s
        end
    end
    r
end

normquantile(a, q1, q2) = normquantile(a, [q1,q2])
function normquantile(a, q = [0.1, 0.9])
    qs = quantile(vec(a), q) 
    clamp((a .- qs[1]) ./ (qs[2]-qs[1]) .* (q[2]-q[1]) .+ q[1], 0, 1)
end

function normunique(a)  # FIXME
    u = unique(a)
    d = Dict()
    [d[u[i]] = i for i = 1:length(u)]
    d
    r = zero(a)
    for i = 1:length(a)
        r[i] = d[a[i]]
    end
    r
end


# #######################################
# ##  minimum and maximum with optional functions and number of values to find

import Base.maximum
maximum_ = maximum
maximum(f::Function, g::Function) = error("not defined")
maximum(x::Array,f::Function) = @p map x f | argmax | at x _

import Base.minimum
minimum_ = minimum
minimum(f::Function, g::Function) = error("not defined")
minimum(x::Array,f::Function) = @p map x f | argmin | at x _

function valuemap(a, m::Dict, default = 0)
    map(x->get(m,x,default),a)
end
function unvaluemap(a, m::Dict, default = 0)
    valuemap(a, flip(m), default)
end

function valuemap(data, mapping)
    r = Array{promote_type(eltype(mapping),(eltype(data))), ndims(data)}(undef,size(data))
    for i = 1:length(data)
        v = data[i]
        if !isa(v,Number) || (!isnan(v) && v>0)
            if isa(v,Number) && v>length(mapping)
                error("valuemap: v>length(mapping), with v==$v and length(mapping)==$(length(mapping))")
            end
            r[i] = mapping[v]
        else
            r[i] = data[i]
        end
    end
    r
end

import Base.clamp, Base.clamp!
clamp(a::AbstractArray, mi::Number, ma::Number) = clamp.(a, mi, ma)
clamp(a::AbstractArray{T}, mi::Union{AbstractArray,Tuple}, ma::Union{AbstractArray,Tuple}) where {T} = (r = Base.copy(a); clamp!(r, r, mi, ma); r)
clamp!(a::AbstractArray{T,2}, mi::Union{AbstractArray,Tuple}, ma::Union{AbstractArray,Tuple}) where {T} = clamp!(a, a, mi, ma)
function clamp!(r::AbstractArray{T,2}, a::AbstractArray{T,2}, mi::Union{AbstractArray,Tuple}, ma::Union{AbstractArray,Tuple}) where {T}
    if !(size(a,1)==length(mi)==length(ma))
        error("clamp!: size(a,1)==length(mi)==length(ma) was false: $(size(a,1))==$(length(mi))==$(length(ma))")
    end
    for j = 1:size(a,2), i = 1:size(a,1)
        r[i,j] = min(max(mi[i],a[i,j]), ma[i])
    end
    r
end
clamp(a::Tuple, mi::Union{AbstractArray,Tuple}, ma::Union{AbstractArray,Tuple}) = (r = Base.copy(a); clamp!(r,r,mi,ma); r)
clamp!(a, mi, ma) = clamp!(a, a, mi, ma)
function clamp!(r, a, mi, ma)
    if !(size(a,1)==length(mi)==length(ma))
        error("clamp!: size(a,1)==length(mi)==length(ma) was false: $(size(a,1))==$(length(mi))==$(length(ma))")
    end
    for i = 1:length(a)
        r[i] = min(max(mi[i],a[i]), ma[i])
    end
    r
end
clamp(a, v::AbstractArray) = clamp(a, ones(ndims(v),1), siz(v))

function nanfunction(f,a::Array{T,2},dim) where {T}
  if dim == 1
    r = NaN*zeros(1,size(a,2))
    for n = 1:size(a,2)
      v = a[:,n]
      v = v[.!isnan.(v)]
      if !isempty(v)
        r[1,n] = f(v)
      end
    end
    return r
  elseif dim == 2
    r = NaN*zeros(size(a,1),1)
    for m = 1:size(a,1)
      v = a[m,:]
      v = v[.!isnan.(v)]
      if !isempty(v)
        r[m,1] = f(v)
      end
    end
    return r
  else
    error("dim must be 1 or 2")
  end
end
nanfunction(f,a) = f(a[.!isnan.(a)])
nanmedian(a) = nanfunction(median,a)
nanmedian(a,dim) = nanfunction(median,a,dim)
nanmean(a) = nanfunction(mean,a)
nanmean(a,dim) = nanfunction(mean,a,dim)
nanstd(a) = nanfunction(std,a)
nanstd(a,dim) = nanfunction(std,a,dim)

# function nanfunction(f,a::Array{T,3},dim) where {T}
#   if dim == 1
#     r = NaN*zeros(1,size(a,2),size(a,3))
#     for n = 1:size(a,2), o = 1:size(a,3)
#       v = a[:,n,o]
#       v = v[!isnan(v)]
#       if !isempty(v)
#         r[1,n,o] = f(v)
#       end
#     end
#     return r
#   elseif dim == 2
#     r = NaN*zeros(size(a,1),1,size(a,3))
#     for m = 1:size(a,1), o = 1:size(a,3)
#       v = a[m,:,o]
#       v = v[!isnan(v)]
#       if !isempty(v)
#         r[m,1,o] = f(v)
#       end
#     end
#     return r
#   elseif dim == 3
#     r = NaN*zeros(size(a,1),size(a,2),1)
#     for m = 1:size(a,1), n = 1:size(a,2)
#       v = a[m,n,:]
#       v = v[!isnan(v)]
#       if !isempty(v)
#         r[m,n,1] = f(v)
#       end
#     end
#     return r
#   else
#     error("dim must be 1, 2 or 3")
#   end
# end
# nanfunction(f,a) = f(a[!isnan(a)])
# nanmedian(a) = nanfunction(median,a)
# nanmedian(a,dim) = nanfunction(median,a,dim)
# nanmean(a) = nanfunction(mean,a)
# nanmean(a,dim) = nanfunction(mean,a,dim)
# nanstd(a) = nanfunction(std,a)
# nanstd(a,dim) = nanfunction(std,a,dim)
# @test_equal nanmedian([1 2 3], 1)  [1 2 3]
# @test_equal nanmedian([1 2 NaN], 1)  [1 2 NaN]
# @test_equal nanmedian([1 2 3], 2)  [2]
# @test_equal nanmedian([1 2 NaN], 2)  [1.5]
# @test_equal nanmedian([1, 2, NaN])  1.5
# d = rand(2,3,4)
# @test_equal nanmean(d,1) mean(d,1)
# @test_equal nanmean(d,2) mean(d,2)
# @test_equal nanmean(d,3) mean(d,3)

mutable struct WhiteningBase
    mean
    base
    vars
end

function zcawhitening(a; kargs...)
    b = flatten(Base.map(col, row(a)))
    r = zcawhitening(b; kargs...)
    s = size(a[1])
    reshape([reshape(r[:,i], s) for i in 1:length(a)], size(a))
end

pcawhitening(a, base = []; kargs...) = zcawhitening(a, base; pcawhitening = true, kargs...)
function zcawhitening(a::Array{T}, base = [];  perpatchnormalization = false, pcawhitening = false, keepvar = 0.95) where {T<:Real}
    # a ... nDim x nSamples
    if perpatchnormalization
        a = a .- mean(a,1)
        s = std(a,1)
        ind = find(s .> 0.)
        a[:,ind] = a[:,ind] ./ s[1,ind]
    end
    if !isa(base, WhiteningBase)
        m = mean(a,2)
        a = a .- m
        sigma = a * a' ./ size(a, 2)
        (U,S,V) = svd(float(sigma))
        coeff = U' * a
        base = WhiteningBase(m, U, S)
    end
    coeff = base.base' * (a .- base.mean)
    S = base.vars
    r = diagm(1 ./sqrt(base.vars .+ 1e-5)) * coeff
    if !pcawhitening
        r = base.base * r
    end
    r, base
end


distance(a) = distance(a,a)
function distance(a::Vector{T},b::Vector{T}) where {T}
    r = zero(eltype(a))
    for i = 1:length(a)
        r += (a[i]-b[i])^2
    end
    sqrt(r)
end

distance(a::AbstractArray{T1,1},b::AbstractArray{T2,2}) where {T1,T2} = distance(col(a),b)
distance(a::AbstractArray{T1,2},b::AbstractArray{T2,1}) where {T1,T2} = distance(a,col(b))

function distance(a::AbstractArray{T1,2},b::AbstractArray{T2,2}) where {T1,T2}
    #% Author   : Roland Bunschoten
    #%            University of Amsterdam
    #%            Intelligent Autonomous Systems (IAS) group
    #%            Kruislaan 403  1098 SJ Amsterdam
    #%            tel.(+31)20-5257524
    #%            bunschot@wins.uva.nl
    #% Last Rev : Oct 29 16:35:48 MET DST 1999
    #% Tested   : PC Matlab v5.2 and Solaris Matlab v5.3
    #% Thanx    : Nikos Vlassis
    #
    #% Copyright notice: You are free to modify, extend and distribute 
    #%    this code granted that the author of the original code is 
    #%    mentioned as the original author of the code.
    #%% modified by eva.dittrich@student.tuwien.ac.at due to 
    #% post by Oliver Woodford (Nov 2008)
    #% on http://www.mathworks.com/matlabcentral/fileexchange/71

    aa = sum(a.*a, dims = 1)
    bb = sum(b.*b, dims = 1)
    sqrt.(abs.((aa' .+ bb) - 2*a'*b))
end
distance(a::Number,b::Number) = abs(a-b)

randbase(basedim, origdim) = randn(MersenneTwister(0), basedim, origdim)
randnormbase(basedim, origdim) = @p randbase basedim origdim | transpose | map normeuclid | transpose

randproj(a, targetdim::Int) = randbase(targetdim, sizem(a))*a

dropdims_(a) = dropdims(a, dims = ndims(a))
mean_(a) = dropdims_(mean(a, ndims(a)))
median_(a) = dropdims_(median(a, ndims(a)))
std_(a) = dropdims_(std(a, ndims(a)))
var_(a) = dropdims_(var(a, ndims(a)))
min_(a) = dropdims_(minimum(a, ndims(a)))
max_(a) = dropdims_(maximum(a, ndims(a)))
sum_(a) = dropdims_(sum(a, ndims(a)))

normmean_(a) = a .- mean_(a)
