export Sampler, sample, sample!, call

immutable Sampler{T,N}
    image::Array{T,N}
    ind::Array{Int,1}
    mi::Array{Int,2}
    ma::Array{Int,2}
    clampedpos::Array{Int,2}
    buf::Array{T,N}
    normmeanstd::Bool
end

function Sampler(image, bsize::Int, scale = 1.; centered::Bool = true, col::Bool = false, normmeanstd::Bool = false)
    patchsize = tuple(repeat(bsize, ndims(image))...)
    if centered
        grid = @p centeredmeshgrid patchsize | times scale | plus 1 |  asint
    else
        grid = @p meshgrid patchsize | times scale | asint
    end
    ind = @p map grid subtoind image | minus 1

    buf = zeros(eltype(image), patchsize)
    if col buf = FunctionalData.col(buf) end
    
    mi = ones(Int,ndims(image),1) - minimum(grid,2) + 1
    ma = siz(image) - maximum(grid,2) + 1
    clampedpos = zeros(Int, size(mi))

    Sampler(image, ind, mi, ma, clampedpos, buf, normmeanstd)
end

call(a::Sampler, pos) = sample(pos, a)
sample(pos, a::Sampler) = sample!(a.buf, pos, a::Sampler)

function sample!(buf, pos::Array{Int,2}, a::Sampler)
    c = @p clamp! a.clampedpos pos a.mi a.ma | subtoind a.image
    for i = 1:length(a.buf)
        buf[i] = a.image[c+a.ind[i]]
    end
    a.normmeanstd && normmeanstd!(buf)
    buf
end


