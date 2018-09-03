export jetcolormap, asimage, asimagesc, asimagescrgb, blocksvisu, pad, image2array, poly2mask, embedvisu
export jetcolors, jetcolorshex, jetcolorants
export aslogimage

function jetcolormap(n)
    step(m, i) = i>=m ? 1.0 : 0.0
    grad(m, i) = (i-m)/(n/4) * step(m, i)

    a = n/8
    f(m, i) = grad(m+3a, i) - grad(m+a, i) - grad(m-a, i) + grad(m-3a, i)

    r = zeros(3,n)
    r[1, :] = [f(3n/4, i) for i in 1:n]
    r[2, :] = [f(n/2, i) for i in 1:n]
    r[3, :] = [f(n/4, i) for i in 1:n]
    clamp(r, 0f0, 1f0)
end
function jetcolors(a,mi,ma, n = 101)
    j = jetcolormap(n)
    @p collect a | minus mi | divby (ma-mi+eps()) | clamp 0. 1. | times 99.9 | plus 1 | round Int _ | part j _
end
jetcolors(a, args...) = jetcolors(a, minimum(a), maximum(a)+0.01)
jetcolorants(a...) = mapvec(jetcolors(a...),x->RGB(x[1],x[2],x[3]))
jetcolorshex(args...) = @p times jetcolors(args...) 255 | round UInt8 _ | map x->@p vec x | bytes2hex | (x->"#$x") 

function asimagescrgb(a, norm = true)
    a = asfloat32(a)
    cm = jetcolormap(256)
    r = Array{Float32,3}(undef, size(a, 1), size(a, 2), 3)
    normf = norm ? norm01 : identity
    b = round(Int, normf(a)*255) .+ 1
    r[:,:,1] = cm[1, b[:]]
    r[:,:,2] = cm[2, b[:]]
    r[:,:,3] = cm[3, b[:]]
    r
end

# import Images
# function asimage(a::AbstractArray{T,2}) where T
#     Images.colorview(Images.Gray, a)
# end

# function asimage(a::AbstractArray{T,3}) where T
#     Images.colorview(Images.RGB, permutedims(a,[3,1,2]))
# end
# asimagesc = asimage*asimagescrgb

function blocksvisu(a, padding = 0)
    a = unstack(a)
    n = len(a)
    typ = eltype(fst(a))
    paddingf(a...) = ones(typ,a...)*padding
    padsize(a) = ceil(Int,a/4)
    try
        a = @p map a reshape | unstack
    catch
    end
    @assert @p mapvec a size | uniq | len | isequal 1
    if ndims(fst(a))==3
        return @p map (1:3) (i->@p map a at i | blocksvisu padding) | stack
    end
    z = @p paddingf padsize(size(fst(a),1)) size(fst(a),2)
    a = @p partsoflen a ceil(Int,sqrt(n)) | map riffle z | map col | map flatten
    z = @p paddingf size(fst(a),1) padsize(size(fst(a),2))
    a[end] = @p pad a[end] siz(fst(a)) padding
    r = @p riffle a z | row | flatten
end

function embedvisu(a::Matrix, patches::AbstractArray{T,N}, s = 2000) where {T<:Number,N}
    mi = minimum(a, dims = 2)
    ma = maximum(a, dims = 2)
    rm = rangem(patches)
    rn = rangen(patches)
    dim = ndims(patches) == 3 ? 1 : 3
    t(a) = round(Int, (a-mi)./(ma-mi) * s)
    r = zeros(Float32, s+sizem(patches),s+sizen(patches),dim)
    for i = 1:len(patches)
        x = @p at a i | t
        p = @p at patches i
        r[x[1] .+ rm, x[2] .+ rn, :] = p
    end
    FD.squeeze(r)
end

function pad(a, siz, value = 0)
    r = value * onessiz(siz, eltype(a))
    if ndims(r) == 1
        r[1:length(a)] = a
    elseif ndims(r) == 2
        r[1:size(a,1),1:size(a,2)] = a
    elseif ndims(r) == 3
        r[1:size(a,1),1:size(a,2),1:size(a,3)] = a
    elseif ndims(r) == 4
        r[1:size(a,1),1:size(a,2),1:size(a,3),1:size(a,4)] = a
    elseif ndims(r) == 5
        r[1:size(a,1),1:size(a,2),1:size(a,3),1:size(a,4),1:size(a,5)] = a
    else
        error("ndims not supperted")
    end
    r
end

image2array(img) = @p map Any[:r,:g,:b] (i->map(y->y.(i), img.data)) | stack

aslogimage(a) = @p clamp a 0 Inf | plus 1 | log | asimagesc



