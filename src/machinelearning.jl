export crossval

crossval(a::Tuple, args...; kargs...) = loocv(a..., args...; kargs...)

function crossval(data, labels, train; predict = predict, n = 4, predictdata = [], shuffle = true,
    nruns = n)
    if !isempty(predictdata) && len(predictdata) != len(data)
        error("len(predictdata)==$(len(predictdata)) is not equal len(data)==$(len(data))")
    end

    ind = shuffle ? randperm(1:len(data)) : collect(1:len(data))

    D = @p part data ind | partition n
    L = @p part labels ind | partition n
    invind = @p invperm ind | partition n

    predictdata = isempty(predictdata) ? D : @p part predictdata ind | partition n

    assert(nruns <= n)
    r = cell(nruns)
    for i in 1:nruns
        d1, _ = cut(D,i)
        _, d2 = cut(predictdata,i)
        l1, l2 = cut(L,i)
        m = train(flatten(d1),flatten(l1))
        r[i] = @p predict m flatten(d2) | part invind[i]
    end
    @p flatten r
end


