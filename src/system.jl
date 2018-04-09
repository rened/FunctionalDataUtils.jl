export ncores, nphysicalcores, systemload, normload, setnumworkers

@static if Sys.isapple() ncores() = asint(read(`sysctl -n hw.ncpu`, String)) end
@static if Sys.isapple() nphysicalcores() = asint(read(`sysctl -n hw.physicalcpu`, String)) end
@static if Sys.islinux() ncores() = asint(read(`nproc --all`, String)) end
@static if Sys.islinux() nphysicalcores() = div(ncores(),2) end

rmcomma(a) = a[end]==',' ? a[1:end-1] : a
systemload() = parsefloat(rmcomma(split(read(`uptime`, String))[end-2]))
normload() = systemload()/ncores()
 
function setnumworkers(n)
    oldn = nprocs()-1
    if n>oldn
        return addprocs(n-oldn)
    elseif n<oldn
        rmthese = workers()[end-(oldn-n)+1:end]
    end
end

