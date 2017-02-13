export @timedone, fasthash, cache, dictcache
export loadedmodules, reloadmodules

macro timedone(a, ex)
    quote
        local t = time()
        println($a, " ... ")
        flush(STDOUT)
        r = $ex
        local elapsed = round((time()-t)*100)/100
        println("    done, $elapsed sec!")
        flush(STDOUT)
        r
    end
end


fasthash{T<:Number}(a::Array{T}) = bytes2hex(eltype(a)!=UInt8 ? sha256(reinterpret(UInt8, vec(a))) : sha256(a))

fasthash(a::Number) = fasthash(Any[[a], "__julianumber"])
if VERSION.minor == 3
    fasthash(a::Char) = fasthash(Any[utf8(string(a)), "__juliachar"])
else
    fasthash(a::Char) = fasthash(Any[string(a), "__juliachar"])
end
fasthash(a::AbstractString) = bytes2hex(sha256(a))
fasthash(a::Symbol) = fasthash(Any[string(a),"__juliasymbol"])
fasthash(a::Tuple) = fasthash(Any[a..., "__juliatuple"])
fasthash(f::Function) = fasthash(Any[string(f), "__juliafunction"])
fasthash(a::UnitRange) = fasthash(Any[a.start, a.stop, "__juliaunitrange"])
fasthash(a::Range) = fasthash(Any[a.start, a.step, a.stop, "__juliarange"])
fasthash(a::FloatRange) = fasthash(Any[a.start, a.step, a.len, a.divisor, "__juliafloatrange"])
fasthash(a::Bool) = fasthash("$(a)_juliabool")
fasthash(a) = fasthash(repr(a))

function fasthash(a::Array)
    if length(a)>100000 && isa(a, DenseArray) && eltype(a)<:Number
        fasthash(Any[a[1:911:end],sum(a)])
    else
        r = Base.map(fasthash, a)
        fasthash(join(r))
    end
end
function fasthash(a::Dict)
    K = collect(keys(a))
    ind = sortperm(Base.map(string, K))
    return fasthash([(k, fasthash(a[k])) for k in K[ind]])
end
fasthash(f::Function, args, version) = fasthash(Any[string(f), args..., version])


cache(f::Function, f2::Function, args...) = error("cache(f,f2,...) not supported")
function cache(f::Function, args...; version = "0 none set", kargs...)
    mkpath("zzz")
    h = hash(fasthash([f; args; version; kargs]))
    filename = "zzz/$(string(f))-$h.jls"
    if existsfile(filename)
        @timedone "reloading $(string(f))" open(deserialize, filename)
    else
        @timedone "computing $(string(f))" begin
            r = f(args...; kargs...)
            open(s->serialize(s,r),filename,"w")
            r
        end
    end
end
cache(a,f::Function,args...; kargs...) = cache(f, Any[a, args...]...; kargs...)


function dictcache(f, args...; version = "0 none set", filepath = "cache.dictfile")
    dictopen(filepath) do cache
        h = fhash(f, args, version)
        key = tuple(string(f), version, h)
        #@show "before" key
        if haskey(cache, key...)
             @timedone "reloading $(string(f))" cache[key...]
        else
            @timedone "computing $(string(f))" begin
              r = f(args...)
              cache[key...] = r
              #@show keys(cache, key[1:end-1])
              assert(in(key[end], keys(cache, key[1:end-1]...)))
              r
            end
        end
        #@show "after" key keys(cache, string(f), version)
    end
end

function loadedmodules()
    a = names(Main)
    r = Any[]
    for x in a
        #try
            if typeof(eval(Main,x))==Module && !in(x,Any[:Main, :Core])
                push!(r, fullname(eval(Main,x))[1])
            end
        #catch e
        #    Base.display_error(e)
        #end
    end
    r
end

function reloadmodules()
    for x in vcat(:__anon__, sort(loadedmodules())) # HACK to make sure HDF5 is loaded before JLD
        if in(x, Any[:MAT_HDF5, :MAT_v5, :IPythonDisplay, :IJulia, :ReactiveContexts])
            try
				a = "@everywhere using $x"
				println("trying: $a")
                eval(Main, parse(a))
            catch e
				showerror(e)
            end
        end
    end
end
