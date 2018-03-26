export ncores, nphysicalcores, systemload, normload, setnumworkers

@static if Sys.isapple() ncores() = asint(readall(`sysctl -n hw.ncpu`)) end
@static if Sys.isapple() nphysicalcores() = asint(readall(`sysctl -n hw.physicalcpu`)) end
@static if Sys.islinux() ncores() = asint(readall(`nproc --all`)) end
@static if Sys.islinux() nphysicalcores() = div(ncores(),2) end

rmcomma(a) = a[end]==',' ? a[1:end-1] : a
systemload() = parsefloat(rmcomma(split(readall(`uptime`))[end-2]))
normload() = systemload()/ncores()
 
function setnumworkers(n)
    oldn = nprocs()-1
    if n>oldn
        return addprocs(n-oldn)
    elseif n<oldn
        rmthese = workers()[end-(oldn-n)+1:end]
    end
end

