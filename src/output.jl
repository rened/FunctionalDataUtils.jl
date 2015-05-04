export disp, showdict, log

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

import Base.log                                              
log(a::Union(String,ASCIIString); kargs...) = log(STDOUT, a; kargs...)
log(io::IO, a::Union(String, ASCIIString); indent = 0) = println(io, Libc.strftime("%Y-%m-%d %T %z %Z", time()), "  |  ", repeat("  ", indent), a)
