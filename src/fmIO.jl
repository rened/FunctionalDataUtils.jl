function fmwrite(a, filename::ASCIIString)

    stream = open(filename,"w");
    dummy = Any[]; finalizer(dummy, (x)->close(stream))

    j2m, m2j = typedictionaries()
    fmwrite_internal(stream, a, j2m)
    close(stream)

end

function fmread(filename::ASCIIString)

    stream = open(filename,"r");
    dummy = Any[]; finalizer(dummy, (x)->close(stream))

    j2m, m2j = typedictionaries()
    r = fmread_internal(stream, m2j)
    close(stream)
    return r
end

function typedictionaries()

    j2m = Dict(Int => "int64", Int8 => "int8", Int16 => "int16", Int32 => "int32", Int64 => "int64", 
        Uint => "uint64", Uint8 => "uint8", Uint16 => "uint16", Uint32 => "uint32", Uint64 => "uint64", 
        Float32 => "single", Float64 => "double", ASCIIString => "char*1", Bool => "logical", Any => "cell", 
        Dict => "struct")

    m2j = Dict{ASCIIString,Type}()

    for (k,v) in j2m
        newv = v * repeat("-", 7-length(v))
        j2m[k] = newv
        m2j[newv] = k
    end

    return j2m, m2j 

end


function fmwrite_internal(stream, a, j2m)

    ###################
    #  write type

    if typeof(a)==Char
        a = string(a)
    end
    if typeof(a)<:Array
        t = eltype(a)
    else
        t = typeof(a)
    end
    if typeof(a)<:Dict
        t = Dict
    end

    write(stream, j2m[t]);

    ###################
    #  write size

    s = zeros(Uint64, 1, 10)

    if typeof(a)==ASCIIString
        s[1:2] = [1 length(a)]
    elseif typeof(a)<:Dict || ndims(a)==0
        s[1:2] = [1 1]
    else 
        if ndims(a)==1
            s[1:2] = [1, size(a,1)];
        else
            for i = 1:ndims(a)
                s[i] = size(a,i)
            end
        end
    end
    if s[1]==0
        s[1] = 1
    end
    if s[2]==0
        s[2] = 1
    end

    #println("s: ",s)

    assert(sum(s)>0);
    write(stream, s);

    ###################
    #  write data

    if typeof(a)<:Dict
        list = Any[]
        for (k,v) in a
            push!(list, [k,v])
        end
        fmwrite_internal(stream, list, j2m)

    elseif t<:Number #|| (typeof(a)<:Array && eltype(a)<:Number)
        write(stream, a)

    elseif t==Any
        for i = 1:length(a)
            fmwrite_internal(stream, a[i], j2m)
        end

    else
        write(stream, a)
    #else
    #    error("how did we get here? typeof(a)=",typeof(a))
    end
end




function fmread_internal(stream, m2j)

    ###################
    #  read type

    t = m2j[ascii(read(stream, Uint8, 7))];

    ###################
    #  read size

    s = read(stream, Uint64, 1, 10)
    s = s[s.!=0]
    println("\n---\ngot here: s ",s," sum(s): ",prod(s),"\n")
    println("type is: $t")
    @show nb_available(stream)
    
    ###################
    #  read data

    if t<:Dict
        list = fmread_internal(stream, m2j)
        r = Dict()
        for i = 1:length(list)
            r[list[i][1]] = list[i][2]
        end

    elseif t==Any
        r = Array(t, s...)
        for i = 1:length(r)
            r[i] = fmread_internal(stream, m2j)
        end

    elseif t==ASCIIString
        r = ascii(read(stream, Uint8, prod(s)))
    else
        r = Array(t, s...)
        read!(stream, r)
        read(stream, r)
    end

    return r
end











