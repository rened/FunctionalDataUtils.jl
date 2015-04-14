using MultivariateStats
export iimg, iimg!
export interp3, interp3with01coords, resize, resizeminmax
export grid, meshgrid, meshgrid3, centeredgrid, centeredmeshgrid, overlaygradient, toranges, tosize, tosize3
export imregionalmin, imregionalmax, monogen, bwlabel, bwlabel!, monoslic, border, sortcoords, bwdist
export blocks, blocks!!
export rle, unrle
export stridedblocksub
export inpolygon, inpointcloud


#######################################
## iimg

iimg(a) = (r = float(a); iimg!(r); return r)

function iimg!{T}(a::AbstractArray{T,2})
    for x = 1:size(a,2)
        for y = 2:size(a,1)
            a[y,x] += a[y-1,x] 
        end
    end

    for y = 1:size(a,1)
        for x = 2:size(a,2)
            a[y,x] += a[y,x-1]
        end
    end
end

function iimg!{T}(a::AbstractArray{T,3})
    for z = 1:size(a,3)
        iimg!(sub(a,1:size(a,1),1:size(a,2),z))
    end

    for x = 1:size(a,2)
        for y = 1:size(a,1)
            for z = 2:size(a,3)
                a[y,x,z] += a[y,x,z-1]
            end
        end
    end
end

function interp3{T}(a::Union(Array{T,2}, Array{T,3}), m_::Float64, n_::Float64, o_::Float64)
  if m_<1 || n_<1 || o_<1 || m_>size(a,1) || n_>size(a,2) || o_>size(a,3) 
    # @show m_ n_ o_ size(a)
    error("interp3: index out of bounds: [$m_, $n_, $o_], size(a) == $(size(a))")
  end
  
  if isnan(m_) || isnan(n_) || isnan(o_)
    # @show m_ n_ o_ size(a)
    error("interp3: index is NaN")
  end
  
  rem(a) = a-trunc(a)
  wm = 1-rem(m_)
  wn = 1-rem(n_)
  wo = 1-rem(o_)
  m = floor(Int,m_)
  M = ceil(Int,m_)
  n = floor(Int,n_)
  N = ceil(Int,n_)
  o = floor(Int,o_)
  O = ceil(Int,o_)
  
#  @show wm wn wo m n o M N O
#@show "got here" wm wn wo m M n N o O size(a)
  meanm!no = wm*a[m,n,o]+(1-wm)*a[M,n,o]
  meanm!No = wm*a[m,N,o]+(1-wm)*a[M,N,o]
  meanm!nO = wm*a[m,n,O]+(1-wm)*a[M,n,O]
  meanm!NO = wm*a[m,N,O]+(1-wm)*a[M,N,O]
  
  meanm!nOm!NO = wn*meanm!nO+(1-wn)*meanm!NO
  meanm!nom!No = wn*meanm!no+(1-wn)*meanm!No
  
  wo*meanm!nom!No + (1-wo)*meanm!nOm!NO
end

interp3with01coords(a,m,n,o) = interp3(a, 1+(size(a,1)-1)*m, 1+(size(a,2)-1)*n, 1+(size(a,3)-1)*o)


#####################################################
##   resize

resize(a, factor::Int) = resize(a, round(factor*siz(a)), nearest = true)
resize(a, factor::Number) = resize(a, round(factor*siz(a)))
function resize{T<:Real}(a::Array{T}, s::Union(AbstractArray); nearest = false)
    if size(a) == tuple(s...)
        return Base.copy(a)
    end
    r = Array(nearest ? eltype(a) : Float32, s...)
    mi = linspace(1., size(a,1), size(r,1))
    ni = linspace(1., size(a,2), size(r,2))
    oi = linspace(1., size(a,3), size(r,3))
    if nearest
        mi = int(mi)
        ni = int(ni)
        oi = int(oi)
        resize_kernel_nearest(r, a, mi, ni, oi)
    else
        resize_kernel_interp3(r, a, mi, ni, oi) 
    end
    r
end
function resize_kernel_nearest(r, a, mi, ni, oi) 
    for o = 1:size(r,3), n = 1:size(r,2), m = 1:size(r,1)
        r[m,n,o] = a[mi[m], ni[n], oi[o]]
    end
end
function resize_kernel_interp3(r, a, mi, ni, oi) 
    for o = 1:size(r,3), n = 1:size(r,2), m = 1:size(r,1)
        r[m,n,o] = interp3(a, mi[m], ni[n], oi[o])
    end
end


#####################################################
##   resizeminmax


function resizeminmax(a, mins, maxs)

    assert(ndims(a)==length(mins))
    assert(ndims(a)==length(maxs))

    s = [size(a)...]

    if Base.any(s .< mins)
        mi = minimum(s ./ mins)
        minewsiz = round(s / mi)
    else
        minewsiz = s
    end

    if Base.any(s .> maxs)
        ma = maximum(s ./ maxs)
        manewsiz = round(s / ma)
    else
        manewsiz = s
    end
    newsiz = round(Int,FunctionalData.clamp((minewsiz + manewsiz) / 2, mins, maxs))
    #@show size(a) newsiz
    resize(a, newsiz)
end


function overlaygradient(img, sp, sp2 = sp*0, sp3 = sp*0)
    if size(img,1)!=size(sp,1) || size(img,2)!=size(sp,2)
        error("overlaygradient: sizes are not equal: size(img) == $(size(img)) and size(sp)==$(size(sp))")
    end
    sp = float(sp)
    sp2 = float(sp2)
    sp3 = float(sp3)
    if size(img,3)>1
        img = mean(img,3)
    end
    img = norm01(img)
    r = cat(3,img,img,img)

    function gradient(a)
        g = falses(size(a))
        for n = 1:size(a,2)-1, m = 1:size(a,1)
            g[m,n] |= a[m,n+1] != a[m,n]
        end
        for n = 1:size(a,2), m = 1:size(a,1)-1
            g[m,n] |= a[m+1,n] !=a [m,n]
        end
        g
    end

    gs = Base.map(gradient, Any[sp, sp2, sp3])

    for n = 1:size(img,2), m = 1:size(img,1)
    r[m,n,1] = gs[1][m,n] ? 1 : r[m,n,1]
    r[m,n,2] = gs[2][m,n] ? 1 : r[m,n,2]
    r[m,n,3] = gs[3][m,n] ? 1 : r[m,n,3]
    end
    r
end

toranges(a) = [a[1,i]:a[2,i] for i in 1:len(a)]
tosize(a) = tuple((a[2,:]-a[1,:].+1)...)
tosize3(a) = (size(a,2)<3 ? a = cat(2,a,[1 2]') : nothing; tosize(a))

# function grid(rm,rn)
#   m = [m for m in rm, n in rn]
#   n = [n for m in rm, n in rn]
#   return (m,n)
# end
# function grid(rm,rn,ro)
#   m = [m for m in rm, n in rn, o in ro]
#   n = [n for m in rm, n in rn, o in ro]
#   o = [o for m in rm, n in rn, o in ro]
#   return (m,n,o)
# end
# grid{T}(a::Array{T,2}) = grid(1:size(a,1), 1:size(a,2))
# grid{T}(a::Array{T,3}) = grid(1:size(a,1), 1:size(a,2), 1:size(a,3))

# function meshgrid(a...)
#   t = grid(a...)
#   r = zeros(length(t),length(t[1]))
#   for i = 1:length(t)
#     r[i,:] = t[i][:]
#   end
#   r
# end
    
meshgrid(a::Array) = meshgrid(size(a)...)
meshgrid(a::Int, v::Array) = meshgrid(repeat([a], ndims(v))...)
meshgrid(sm::Int, sn::Int) = meshgrid(1:sm, 1:sn)
meshgrid(sm::Int, sn::Int, so::Int) = meshgrid(1:sm, 1:sn, 1:so)
meshgrid(rm::AbstractArray, rn::AbstractArray) = [row([m for m in rm, n in rn]); row([n for m in rm, n in rn])]
meshgrid(rm::AbstractArray, rn::AbstractArray, ro::AbstractArray) = [row([m for m in rm, n in rn, o in ro]); row([n for m in rm, n in rn, o in ro]); row([o for m in rm, n in rn, o in ro])]

meshgrid3{T}(a::Array{T,2}) = meshgrid(reshape(a, size(a,1), size(a,2), 1))
meshgrid3{T}(a::Array{T,3}) = meshgrid(a)

centeredmeshgrid(a...) = let r = meshgrid(a...); ceil(Int, r .- mean(r,2)) end

# centeredgrid{T}(v::Array{T,2}) = grid((1:size(v,1))-size(v,1)/2, (1:size(v,2))-size(v,2)/2)
# centeredgrid{T}(v::Array{T,3}) = grid((1:size(v,1))-size(v,1)/2, (1:size(v,2))-size(v,2)/2, (1:size(v,3))-size(v,3)/2)
# centeredmeshgrid{T}(v::Array{T,2}) = meshgrid((1:size(v,1))-size(v,1)/2, (1:size(v,2))-size(v,2)/2)
# centeredmeshgrid{T}(v::Array{T,3}) = meshgrid((1:size(v,1))-size(v,1)/2, (1:size(v,2))-size(v,2)/2, (1:size(v,3))-size(v,3)/2)


function imregionalmin{T}(img::Array{T,2})
  r = falses(size(img))
  for n = 2:size(img,2)-1, m = 2:size(img,1)-1
    v = img[m,n]
    if v<img[m-1,n] && v<img[m+1,n] && v<img[m,n-1] && v<img[m,n+1]
      r[m,n] = true
    end
  end
  r
end
function imregionalmin{T}(img::Array{T,3})
  r = falses(size(img))
  for o = 2:size(img,3)-1, n = 2:size(img,2)-1, m = 2:size(img,1)-1
    v = img[m,n,o]
    if v<img[m-1,n,o] && v<img[m+1,n,o] && v<img[m,n-1,o] && v<img[m,n+1,o] && v<img[m,n,o-1] && v<img[m,n,o+1]
      r[m,n,o] = true
    end
  end
  r
end;
imregionalmax(img) = imregionmin(-float(img))




monogen(img, spacing) = monogen_(float32(img), float32(spacing*2))

function monogen_(img::Array{Float32,2}, wavelength::Float32)
  Base.FFTW.set_num_threads(nphysicalcores())
  
  myfft = fft
  myifft(x) = real(ifft(x))

  img = myfft(img)

  rows = size(img,1)
  cols = size(img,2)
  
  # Generate horizontal and vertical frequency grids that vary from
  (u2, u1) = grid([linspace(0.,0.5,ifloor(rows/2)), linspace(-0.5,0.,ceil(Int,rows/2))],
  [linspace(0.,0.5,ifloor(cols/2)), linspace(-0.5,.0,ceil(Int,cols/2))])

  radius = sqrt(u1.^2 + u2.^2)    # Matrix values contain frequency
  radius[1,1] = 1
  radius[end,1] = 1
  radius[1,end] = 1
  radius[end,end] = 1

  H1 = complex64(0,1)*u1./radius   # The two monogenic filters in the frequency domain
  u1 = nothing
  H2 = complex64(0,1)*u2./radius
  u2 = nothing

  logGabor = exp((-(log(radius*wavelength)).^2) / (2 * log(float32(0.65)).^2))
  #radius = nothing
  logGabor[1,1] = 0                    #% undo the radius fudge.
  logGabor[end,1] = 0
  logGabor[1,end] = 0
  logGabor[end,end] = 0

  img = img.*logGabor
  logGabor=nothing
  h = myifft(img.*H1).^2
  H1=nothing
  psi = atan2(myifft(img),sqrt(h+myifft(img.*H2).^2))
  psi = atan2(myifft(img),sqrt(h+myifft(img.*H2).^2))
  h = nothing
  H2 = nothing
  psi = sign(psi).*exp(-abs(psi))
end





function monogen_(img::Array{Float32,3}, wavelength::Float32)
  if size(img,3)==1
    return monogen_(img[:,:,1], wavelength)
  end

  Base.FFTW.set_num_threads(nphysicalcores())

  myfft = fft
  myifft(x) = real(ifft(x))

  img = myfft(img)
  (rows,cols,channels) = size(img)
  (u2, u1, u3) = grid([linspace(0.,0.5,ifloor(rows/2)), linspace(-0.5,0.,ceil(Int,rows/2))],[linspace(0.,0.5,ifloor(cols/2)), linspace(-0.5,0.,ceil(Int,cols/2))],
  [linspace(0.,0.5,ifloor(channels/2)), linspace(-0.5,0.,ceil(Int,channels/2))]);

   radius = sqrt(u1.^2 + u2.^2 + u3.^2);    # Matrix values contain frequency
    # values as a radius from centre
    # (but quadrant shifted)
    
    # Get rid of the 0 radius value in the middle (at top left corner after
    # fftshifting) so that taking the log of the radius, or dividing by the
    # radius, will not cause trouble.
  radius[1,1,1] = 1;
    radius[1,1,end] = 1;
    radius[1,end,1] = 1;
    radius[1,end,end] = 1;
    radius[end,1,1] = 1;
    radius[end,1,end] = 1;
    radius[end,end,1] = 1;
    radius[end,end,end] = 1;
    H1 = u1./radius;   # The three monogenic filters in the frequency domain
  u1=nothing
    H2 = u2./radius;
  u2=nothing
    H3 = u3./radius;
  u3=nothing
    logGabor = exp((-(log(radius*wavelength)).^2) / (2 * float32(log(0.65))^2));
  radius=nothing
  logGabor[1,1,1] = 0;                    # undo the radius fudge.
  logGabor[1,1,end] = 0;               
  logGabor[1,end,1] = 0;               
  logGabor[1,end,end] = 0;             
  logGabor[end,1,1] = 0;               
  logGabor[end,1,end] = 0;             
  logGabor[end,end,1] = 0;             
  logGabor[end,end,end] = 0;           
  img = img.*logGabor;
  logGabor=nothing
  h = myifft(img.*(complex64(0,1)*H1)).^2;
  H1=nothing
  h += myifft(img.*(complex64(0,1)*H2)).^2;
  H2=nothing
  h += myifft(img.*(complex64(0,1)*H3)).^2;
  H3=nothing
    
  psi = atan2(myifft(img), sqrt(h));
  img = nothing
    psi = sign(psi).*exp(-abs(psi));
end

bwlabel(img, args...) = (a = Base.copy(img); bwlabel!(a, args...); a)
bwlabel!(img) = bwlabel!(img, 1)
function bwlabel!(img, startlabel)
    seeds = Any[]
    labelind = round(Int, startlabel)-1
    for o = 1:size(img,3), n = 1:size(img,2), m = 1:size(img,1)
        if img[m,n,o]==0
            push!(seeds,[m,n,o])
            labelind += 1
            #@show seeds
            while length(seeds)>0
                sm,sn,so = pop!(seeds)
                if img[sm,sn,so] == 0
                    img[sm,sn,so] = labelind
                    for oo = -1:1, on = -1:1, om = -1:1
                        m2 = clamp(sm+om, 1, size(img,1))
                        n2 = clamp(sn+on, 1, size(img,2))
                        o2 = clamp(so+oo, 1, size(img,3))
                        if img[m2,n2,o2] == 0
                            push!(seeds,[m2,n2,o2])
                        end
                    end
                end
            end
        end
    end
end


@inbounds function monoslic_kernel(m,n,o,oi2s, ni2s, mi2s, lookup, newcenters, sv, labels)
  bestd = typemax(Float32)
    for oi2 = oi2s 
      for ni2 = ni2s
          for mi2 = mi2s
              if lookup[mi2,ni2,oi2] > 0f0 #
                  diffm = m-newcenters[1,mi2,ni2,oi2]::Float32
                  diffn = n-newcenters[2,mi2,ni2,oi2]::Float32
                  diffo = o-newcenters[3,mi2,ni2,oi2]::Float32
                  d = (diffm*diffm + diffn*diffn + diffo*diffo)::Float32
                  if d < bestd
                      bestd = d
                      sv[m,n,o] = labels[mi2,ni2,oi2]
                  end
              end
          end
      end
  end
  nothing
end     

@inbounds function processslice(o, B, clampedois, clampednis, clampedmis, newcenters, sv, labels, signs, notsigns)
    Bind = size(B,1)*size(B,2)*(o-1)::Int
    for n = 1:size(B,2) 
        for m = 1:size(B,1) 
            Bind += 1
            @inbounds b = B[Bind]
            lookup = b ? signs : notsigns

            monoslic_kernel(m,n,o, clampedois[o], clampednis[n], clampedmis[m], lookup, newcenters, sv, labels)
        end
    end
end


#monoslic(img, spacing; kargs...) = monoslic(float32(img), float32(spacing); kargs...)
#function monoslic(img::Array{Float32}, spacing::Float32; workers = Base.workers())
monoslic(img, spacing; kargs...) = monoslic(float32(img), float32(spacing))
function monoslic(img::Array{Float32}, spacing::Float32)
    println("computing monogen")
    @time M = monogen(img,spacing) 

    if size(M,3) > 1
        oind = round(spacing/2):spacing:size(M,3)
    else
        oind = 1
    end
    
    mind = round(spacing/2):spacing:size(M,1)
    nind = round(spacing/2):spacing:size(M,2)
    centers  = zeros(Float32, 3, length(mind), length(nind), length(oind))
    labels   = zeros(Float32, length(mind), length(nind), length(oind))
    signs    = zeros(Float32, length(mind), length(nind), length(oind))
    bestdist = Inf32*ones(Float32, length(mind), length(nind), length(oind))
    bestind  = zeros(Float32, length(mind), length(nind), length(oind))

    label = 1;
    for o = 1:length(oind), n = 1:length(nind), m = 1:length(mind)
        centers[:,m,n,o] = [mind[m];nind[n];oind[o]]
        labels[m,n,o] = label
        label += 1
    end
    
    labels[:] = randperm(row(labels))

    cand = imregionalmin(abs(M))
    assert(size(cand)==size(M))
    cand = find(cand)

    newcenters = zeros(Float32, size(centers))
    (_, csm, csn, cso) = size(newcenters)

    myind2sub{T}(a::Array{T,2},i) = ((m,n)=ind2sub(size(a),i);(m,n,1))
    myind2sub(a,i) = ind2sub(size(a),i)
    for c = cand
        (m,n,o) = myind2sub(M,c)
        mi = round(Int, m/spacing+0.5)
        ni = round(Int, n/spacing+0.5)
        oi = round(Int, o/spacing+0.5)
        mi = min(mi, csm)
        ni = min(ni, csn)
        oi = min(oi, cso)
        d = (centers[1,mi,ni,oi]-m).^2 +
        (centers[2,mi,ni,oi]-n).^2 + (centers[3,mi,ni,oi]-o).^2
        if d<bestdist[mi,ni,oi]
            bestdist[mi,ni,oi] = d
            bestind[mi,ni,oi] = c
            newcenters[:,mi,ni,oi] = [m;n;o]
            signs[mi,ni,oi] = float32(M[m,n,o] > 0)
        end
    end

    B = M.>0
    M = nothing
    notsigns = abs(signs.-1f0)

    maxd = typemax(eltype(newcenters))::Float32

    sv = zeros(Float32, size(B))
    oi2s = Array(Int,3)
    ni2s = Array(Int,3)
    mi2s = Array(Int,3)
    println("B loop")
    lookup = signs
    mis = int([1:size(B,1)]./spacing.+0.5f0)
    nis = int([1:size(B,2)]./spacing.+0.5f0)
    ois = int([1:size(B,3)]./spacing.+0.5f0)
    clampedmis = [[clamp(mi-1,1,csm), clamp(mi,1,csm), clamp(mi+1,1,csm)] for mi in mis]
    clampednis = [[clamp(ni-1,1,csn), clamp(ni,1,csn), clamp(ni+1,1,csn)] for ni in nis]
    clampedois = [[clamp(oi-1,1,cso), clamp(oi,1,cso), clamp(oi+1,1,cso)] for oi in ois]
    Bind = 0

    @time for o = 1:size(B,3)
        processslice(o, B, clampedois, clampednis, clampedmis, newcenters, sv, labels, signs, notsigns)
    end
    bwlabel!(sv, maximum(sv)+1)
    uind = unique(sv)
    labels = zeros(1,int(maximum(uind)))
    labels[uind] = randperm(1:length(uind))
    int(labels[sv])
end

border{T}(a::Array{T,2}) = @p border cat(3,a,a,a) | snd
function border{T}(a::Array{T,3})
    offsets = @p row [[m n o]'+1 for m in -1:1, n in -1:1, o in -1:1] | flatten | map subtoind a | (-) 1
    r = zeros(a)
    for ind = subtoind(2*ones(siz(a)),a):subtoind(siz(a)-1,a)
        a[ind] == 0 && continue
        isborder = false
        for i = 1:len(offsets)
            if a[ind+offsets[i]] == 0
                isborder = true
                break
            end
        end
        r[ind] = isborder
    end
    r
end

function sortcoords(coords)
    if sizem(coords)!=2
        error("FunctionalDataUtils.sortcoords: only works for 2D contours")
    end
    r = zeros(coords)
    setat!(r,1,fst(coords))
    nope = fill(typemax(eltype(coords)),size(fst(r)))
    setat!(coords, 1, nope)

    for i = 1:len(coords)-1
        ind = @p distance at(r,i) coords | indmin
        setat!(r,i+1, at(coords,ind))
        setat!(coords, ind, nope)
    end
    r
end

bwdist(a::Array) = @p bwdist findsub(a) meshgrid(a) | reshape siz(a)
function bwdist(a, pos) 
    if size(a,1) == size(pos,1)
        @p map pos x->minimum(distance(x,a))
    else
        bwdist(findsub(a), pos)
    end
end

function blocks{T1,T2}(pos::Array{T1,2}, a::Array{T2,2}; blocksize = 32,
    scale = 1,
    grid = round(Int, scale .* centeredmeshgrid(repeat(blocksize, ndims(a))...)),
    borderstyle = :staysinside, precompute = false)
    r = similar(a, blocksize^sizem(pos), len(pos))
    assert(sizem(pos) == ndims(a))

    inds = @p map grid+1 subtoind a | minus 1 | vec

    maxoffset = maximum(abs(grid))
    mi = ones(sizem(pos),1)*maxoffset+1
    ma = siz(a)-maxoffset
    blocks!!(r, copy(pos), a, inds, mi, ma, borderstyle)
    precompute ? (r, (inds, mi, ma, borderstyle)) : r
end

function blocks!!{T}(r::Matrix{T}, pos::Matrix{Int}, a::Matrix{T}, inds::Array{Int}, mi, ma, borderstyle)
    if borderstyle == :staysinside
        pos = @p clamp! pos mi ma
        blockssample_internal!(r,pos,a,inds)
    else
        error("unknown borderstyle '$borderstyle'")
    end
    r
end

function blockssample_internal!{T}(r, pos::Array{Int,2}, a::Array{T,2}, inds::Array{Int,1})
    for n = 1:len(pos)
        posind = sizem(a)*(pos[2,n]-1) + pos[1,n]
        for m = 1:length(inds)
            r[m,n] = a[inds[m]+posind]
        end
    end
end

function rle(a)
    r = Dict()
    r[:size] = size(a)
    counters = zeros(Int, length(a))
    data = zeros(eltype(a), length(a))

    value = a[1]
    counter = 0
    dcounter = 1

    for i = 1:length(a)
        if a[i]!=value 
            data[dcounter] = value
            counters[dcounter] = counter

            value = a[i]
            counter = 1
            dcounter += 1
        else
            counter += 1
        end
    end
    data[dcounter] = value
    counters[dcounter] = counter

    r[:counters] = counters[1:dcounter]
    r[:data] = data[1:dcounter]
    r
end

function unrle(a::Dict)
    data = a[:data]
    counters = a[:counters]
    r = zeros(eltype(data), a[:size])

    rind = 1
    for ind = 1:length(counters)
        value = data[ind]
        for i = 1:counters[ind]
            r[rind] = value
            rind += 1
        end
    end
    r
end

stridedblocksub{T}(a::Array{T,2}, blocksize::Number, stride = blocksize; kargs...) = stridedblocksub(a, blocksize*ones(2,1), stride; kargs...)
function stridedblocksub{T}(a::Array{T,2}, blocksiz, stride = blocksiz; keepshape = false)
    if length(stride) == 1
        stride = stride * ones(2,1)
    end
    sm, sn = size(a)
    rm = 0:blocksiz[1]-1
    rn = 0:blocksiz[2]-1
    r = [(m+rm,n+rn) for m in 1:stride[1]:sm-blocksiz[1]+1, n in 1:stride[2]:sn-blocksiz[2]+1]
    if length(r) == 0
        @show size(a) blocksiz stride
        error("length(r) == 0")
    end
    keepshape ? r : row(r)
end
stridedblocksub{T}(a::Array{T,3}, blocksize::Number, stride = blocksize; kargs...) = stridedblocksub(a, blocksize*ones(3,1); kargs...)
function stridedblocksub{T}(a::Array{T,3}, blocksiz, stride = blocksiz; keepshape = false)
    if length(stride) == 1
        stride = stride * ones(3,1)
    end
    sm, sn, so = size(a)
    rm = 0:blocksiz[1]-1
    rn = 0:blocksiz[2]-1
    ro = 0:blocksiz[3]-1
    r = [(m+rm,n+rn,o+ro) for m in 1:stride[1]:sm-blocksiz[1]+1, n in 1:stride[2]:sn-blocksiz[2]+1, o in 1:stride[3]:so-blocksiz[3]+1]
    assert(length(r)>0)
    keepshape ? r : row(r)
end

inpolygon(point, polygon) = inpolygon(point[1], point[2], polygon)
function inpolygon(m::Int, n::Int, polygon)
    j = len(polygon)
    oddnodes = false
    M = polygon[1,:]
    N = polygon[2,:]

    for i in 1:len(polygon)
        if M[i] < m && M[j] >= m || M[j] < m && M[i] >= m
            if N[i] + (m-M[i]) / (M[j]-M[i]) * (N[j]-N[i]) < n
                oddnodes = !oddnodes
            end
        end
        j = i
    end

    oddnodes
end
poly2mask{T}(polygon::Array{T,2},m::Int,n::Int) = poly2mask(polygon, 1:m, 1:n)
poly2mask{T}(polygon::Array{T,2},img::Array{T,2}) = poly2mask(polygon, 1:size(img,1), 1:size(img,2))
poly2mask{T}(polygon::Array{T,2},M::AbstractArray,N::AbstractArray) = Float32[inpolygon(m,n,polygon) for m in M, n in N]
 
function inpointcloud(point, cloud, n = 8)
    nearest = at(cloud, indmin(distance(point,cloud)))
    #return distance(nearest,point)
    around = @p sortperm vec(distance(nearest,cloud)) | part cloud _ | take n
    pca = fit(PCA, around.+0.0001*randn(size(around)), pratio = 1., method = :svd)
    line = @p last projection(pca)
    #@show projection(pca)
    dists = distance(around, around)
    diameter = maximum(dists)
    relative = cloud.-nearest
    proj = line * (line' * relative)
    dist2line = sqrt(sum((proj-relative).^2,1))
    inpipe = @p part cloud find(dist2line.<=diameter/2)
    cloud2point = point - nearest
    a = sum(distance(nearest + cloud2point, inpipe))
    b = sum(distance(nearest - cloud2point, inpipe))
    #@show nearest cloud2point around dist2line line diameter inpipe a b size(inpipe)
    sum(distance(nearest + cloud2point, inpipe)) < sum(distance(nearest - cloud2point, inpipe))
end



