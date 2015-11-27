export disp, showdict, log, setfilelogging, setlogfile, errorstring

disp(x...) = println(join(map(string,x),", "))

function showdict(a, indent = "")
    sk = sort(collect(keys(a)))
    for k in sk
        print("$(indent)$k: ")
        if isa(a[k], Dict)
            println("")
            showdict(a[k], indent*"  ")
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
log(a::AbstractString; kargs...) = log(STDOUT, a; kargs...)
function log(io::IO, a::AbstractString; indent = 0, tofile = [], toSTDOUT = true)
    buf = IOBuffer()
    println(buf, Libc.strftime("%Y-%m-%d %T %z %Z", time()), "  |  ", repeat("  ", indent), a)
    str = takebuf_string(buf)

    toSTDOUT && println(io, str)
    if (tofile == [] && LOGTOFILE) || tofile == true
        open(LOGFILE, "a") do fid
            write(fid, str)
        end
    end
    nothing
end

function errorstring(e)
    buf = IOBuffer()
    showerror(buf, e, catch_backtrace())
    takebuf_string(buf)
end
