export exitwithteststatus, asfloat32, asfloat64, asint, histdict

function exitwithteststatus()
    s = FactCheck.getstats()
    if s["nNonSuccessful"] == 0
        print("   PASSED!")
    else
        print("   FAILED: $(s["nFailures"]) failures and $(s["nErrors"]) errors")
    end
    println("   ( $(s["nNonSuccessful"]+s["nSuccesses"]) tests for runtests.jl $(join(ARGS, " ")))")
    exit(s["nNonSuccessful"])
end

if VERSION.minor > 3
    asint(a) = round(Int, a)
    asfloat32(a) = map(Float32, a)
    asfloat64(a) = map(Float64, a)
else
    asint = Base.int
    asfloat32 = Base.float32
    asfloat64 = Base.float64
end

function histdict(a, field)
    d = Dict()
    @p map a x->d[x[field]] = get(d,x[field],0) + 1
    d
end
