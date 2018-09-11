export disp, showdict, log, @log, setfilelogging, setlogfile, errorstring, savemv

disp(x...) = println(join(map(string,x),", "))


showdict(a::AbstractDict, args...; kargs...) = showdict(stdout, a, args..., kargs...)
function showdict(io::IO, a::AbstractDict, desc = ""; indent::Int = 0)
    nindent = 2
    if !isempty(desc)
        println(io, desc*":")
        indent += nindent
    end
    sk = sort(collect(keys(a)))
    for k in sk
        print(io, repeat(" ", indent),"$k: ")
        if isa(a[k], Dict)
            println(io, "")
            showdict(io, a[k], indent = indent + nindent)
        else
            println(io, isnil(a[k]) ? repr(a[k]) : "$(a[k])")
        end
    end
end

function setfilelogging(a::Bool)
    global LOGTOFILE
    LOGTOFILE = a
end

function setlogfile(a::AbstractString)
    global LOGFILE
    LOGFILE = a
end

import Base.log                                              
log(a::AbstractString, args...; kargs...) = log(stdout, a, args...; kargs...)
function log(io::IO, a::AbstractString, args...; indent = 0, tofile = [], toSTDOUT = true)
    buf = IOBuffer()
    println(buf, Libc.strftime("%Y-%m-%d %T %z %Z", time()), "  |  ", repeat("  ", indent), join([a, args...], " "))
    str = String(take!(buf))

    toSTDOUT && println(io, str[end] == '\n' ? str[1:end-1] : str)
    if (tofile == [] && LOGTOFILE) || tofile == true || isa(tofile, AbstractString)
        filename = isa(tofile, AbstractString) ? tofile : LOGFILE
        @spawnat 1 open(filename, "a") do fid
            write(fid, str)
        end
    end
    nothing
end

macro log(a...)
    quote
        log($(a...))
    end
end

function errorstring(e)
    buf = IOBuffer()
    showerror(buf, e, catch_backtrace())
    String(take!(buf))
end

savemv(a, filename::String, f::FD.Callable, args...) = savemv(a, f, filename, args...)
function savemv(a, f::FD.Callable, filename::String, args...)
    tmpfilename = @p concat filename "." randstring(20) ".savemv.temp"
    f(a, tmpfilename, args...)
    mv(tmpfilename, filename, remove_destination = true)
    filename
end
