export ncores, nphysicalcores, systemload, normload, setnumworkers, @timedone

@osx_only ncores() = asint(readall(`sysctl -n hw.ncpu`))
@osx_only nphysicalcores() = asint(readall(`sysctl -n hw.physicalcpu`))
@linux_only ncores() = asint(readall(`nproc --all`))
@linux_only nphysicalcores() = ncores()/2
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

