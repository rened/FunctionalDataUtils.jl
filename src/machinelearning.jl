export crossval

crossval(a::Tuple, args...; kargs...) = loocv(a..., args...; kargs...)

function crossval(data, labels, train; predict = predict, n = 4, predictdata = [], shuffle = true)
    if !isempty(predictdata) && len(predictdata) != len(data)
        error("len(predictdata)==$(len(predictdata)) is not equal len(data)==$(len(data))")
    end

    ind = shuffle ? randperm(1:len(data)) : collect(1:len(data))
    invind = invperm(ind)

    D = @p part data ind | partition n
    L = @p part labels ind | partition n

    predictdata = isempty(predictdata) ? D : @p part predictdata ind | partition n

    r = cell(n)
    for i in 1:n
        d1, _ = cut(D,i)
        _, d2 = cut(predictdata,i)
        l1, l2 = cut(L,i)
        m = train(flatten(d1),flatten(l1))
        r[i] = predict(m,flatten(d2))
    end
    @p flatten r | part invind
end


