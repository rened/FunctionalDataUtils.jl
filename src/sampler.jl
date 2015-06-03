export Sampler, sample, sample!, call

immutable Sampler{T,N}
    buf::Array{T,N}
    ind::Array{Int}
    image::Array{T,N}
end

function Sampler(image, bsize::Int, scale = 1.; col = false)
    grid = centeredmeshgrid(repeat(bsize, ndims(image))...)
    ind = @p times grid scale | round Int _ | map subtoind image
    # Sampler{eltype(image),2}(zeros(eltype(image), bsize, bsize), ind)
    buf = zeros(eltype(image), repeat(bsize, ndims(image))...)
    if col
        buf = FunctionalData.col(buf)
    end
    Sampler(buf, ind, image)
end

sample(pos, a::Sampler) = sample!(a.buf, pos, a::Sampler)

function sample!(buf, pos, a::Sampler)
    c = @p subtoind pos a.image
    @inbounds if c+a.ind[1] < 1 || c+a.ind[end] > length(a.image)
        for i = 1:length(buf)
            ind = c+a.ind[i]
            buf[i] = a.image[max(min(ind,length(a.image)),1)]
        end
    else
        for i = 1:length(a.buf)
            buf[i] = a.image[c+a.ind[i]]
        end
    end
    buf
end

call(a::Sampler, pos) = sample(pos, a)

