export exitwithteststatus, asfloat16, asfloat32, asfloat64, asint, histdict
export tryasfloat32, tryasfloat64, tryasint
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

