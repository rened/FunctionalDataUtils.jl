export ncores, nphysicalcores, systemload, normload, setnumworkers, @timedone

@static if is_apple() ncores() = asint(readall(`sysctl -n hw.ncpu`)) end
@static if is_apple() nphysicalcores() = asint(readall(`sysctl -n hw.physicalcpu`)) end
@static if is_linux() ncores() = asint(readall(`nproc --all`)) end
@static if is_linux() nphysicalcores() = div(ncores(),2) end

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

