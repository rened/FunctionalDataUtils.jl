export exitwithteststatus, asfloat16, asfloat32, asfloat64, asint, histdict, asdict
export tryasfloat32, tryasfloat64, tryasint, fromimage
export serve

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

asint(a) = round(Int, a)
asint(a::AbstractString) = parse(Int, a)
asfloat16(a) = map(Float16, a)
asfloat32(a) = map(Float32, a)
asfloat32(a::AbstractString) = parse(Float32, a)
if isinstalled("Images")
    function asfloat32{T<:AbstractImage}(a::T)
        a = raw(a)
        if ndims(a) == 3
            a = permutedims(a,[3,2,1])
            a = a[:,:,1:min(sizeo(a),3)]
        else
            a = a'
        end
        @p asfloat32 a | divby 255 | clamp 0 1
    end
end

asfloat64(a) = map(Float64, a)
asfloat64(a::AbstractString) = parse(Float64, a)

tryasint(a, d = a) = try asint(a) catch return d end
tryasfloat32(a, d = a) = try asfloat32(a) catch return d end
tryasfloat64(a, d = a) = try asfloat64(a) catch return d end

histdict(a, field) = @p extract a field | histdict
function histdict(a)
    d = Dict()
    @p map a x->d[x] = get(d,x,0) + 1
    d
end

asdict(a) = @p fieldnames a | map (x->Pair(x,getfield(a,x))) | Dict
asdict{T<:Union{Dict,AbstractString,Array}}(a::T) = a

function serve(basepath::AbstractString, args...;kargs...)
    basepath = abspath(basepath)
    serve(args...; kargs...) do req, res
        filename = @p concat basepath req.resource | normpath
        @show filename
        if startswith(filename, basepath) && existsfile(filename) && !isdir(filename)
            HttpServer.FileResponse(filename)
        else
            Response(403)
        end
    end
end

function serve(f::Function, portrange = 8080:8199; ngrok = false)
    eval(:(using HttpServer))
    server = Server(HttpHandler(f))
    port, tcp = listenany(fst(portrange))
    close(tcp)
    @async run(server, port)
    if ngrok
        (stdout,stdin,process) = readandwrite(`ngrok http $port -log stdout --log-level "debug"`);
        a = eachline(stdout)
        b = []
        @async for x in a
            if contains(x,"Hostname:")
                push!(b,x)
                break
            end
        end
        for x in 1:50
            isempty(b) || break
            sleep(0.1)
        end
        return @p fst b | split "Hostname:" | last | split ".ngrok.io" | fst | concat ".ngrok.io"
    end
    Int(port)
end

