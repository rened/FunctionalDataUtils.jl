export disp, showdict, log, @log, setfilelogging, setlogfile, errorstring

disp(x...) = println(join(map(string,x),", "))

function showdict(a, desc = ""; indent::Int = 0)
    const nindent = 2
    if !isempty(desc)
        println(desc*":")
        indent += nindent
    end
    sk = sort(collect(keys(a)))
    for k in sk
        print(repeat(" ", indent),"$k: ")
        if isa(a[k], Dict)
            println("")
            showdict(a[k], indent = indent + nindent)
        else
            println("$(a[k])")
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
log(a::AbstractString, args...; kargs...) = log(STDOUT, a, args...; kargs...)
function log(io::IO, a::AbstractString, args...; indent = 0, tofile = [], toSTDOUT = true)
    buf = IOBuffer()
    println(buf, Libc.strftime("%Y-%m-%d %T %z %Z", time()), "  |  ", repeat("  ", indent), join([a, args...], " "))
    str = takebuf_string(buf)

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
    takebuf_string(buf)
end
