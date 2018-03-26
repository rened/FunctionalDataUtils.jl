export makeSampler, Sampler, sample, sample!, call

mutable struct Sampler
    image::Array{Float32,2}
    ind::Array{Int,1}
    mi::Array{Int,2}
    ma::Array{Int,2}
    clampedpos::Array{Int,2}
    buf::Array{Float32,2}
    normmeanstd::Bool
end

function makeSampler(image, bsize::Int, scale = 1.; centered::Bool = true, col::Bool = false, normmeanstd::Bool = false)
    patchsize = tuple(repeat(bsize, ndims(image))...)
    if centered
        grid = @p centeredmeshgrid patchsize | times scale | plus 1 |  asint
    else
        grid = @p meshgrid patchsize | times scale | asint
    end
    ind = @p map grid subtoind image | minus 1

    buf = zeros(eltype(image), patchsize)
    if col buf = FunctionalData.col(buf) end
    
    mi = ones(Int,ndims(image),1) .- minimum(grid, dims = 2) .+ 1
    ma = siz(image) .- maximum(grid, dims = 2) .+ 1
    clampedpos = zeros(Int, size(mi))

    Sampler(image, ind, mi, ma, clampedpos, buf, normmeanstd)
end

(a::Sampler)(pos) = sample(pos, a)

sample(a::Sampler,b::Sampler) = error("Only one of the params can be a sampler.")
sample(pos::AbstractArray{T,2}, a::Sampler) where T<:Integer = sample!(a.buf, pos, a::Sampler)
sample(a::Sampler, pos::AbstractArray{T,2}) where T<:Integer = sample!(a.buf, pos, a::Sampler)

function sample!(buf, pos::AbstractArray{Int,2}, a::Sampler)
    c = @p clamp! a.clampedpos pos a.mi a.ma | subtoind a.image
    for i = 1:length(a.buf)
        buf[i] = a.image[c+a.ind[i]]
    end
    a.normmeanstd && normmeanstd!(buf)
    buf
end


