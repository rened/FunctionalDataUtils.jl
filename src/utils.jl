export exitwithteststatus, asfloat16, asfloat32, asfloat64, asint, histdict
export tryasfloat32, tryasfloat64, tryasint

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
    asint(a::AbstractString) = parse(Int, a)
    asfloat16(a) = map(Float16, a)
    asfloat32(a) = map(Float32, a)
    asfloat32(a::AbstractString) = parse(Float32, a)
    asfloat64(a) = map(Float64, a)
    asfloat64(a::AbstractString) = parse(Float64, a)
else
    asint = Base.int
    asfloat32 = Base.float32
    asfloat64 = Base.float64
end

tryasint(a, d = -1) = try asint(a) catch return d end
tryasfloat32(a, d = NaN) = try asfloat32(a) catch return d end
tryasfloat64(a, d = NaN) = try asfloat64(a) catch return d end

histdict(a, field) = @p extract a field | histdict
function histdict(a)
    d = Dict()
    @p map a x->d[x] = get(d,x,0) + 1
    d
end
