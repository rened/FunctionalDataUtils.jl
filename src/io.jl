using DictFiles
macro snapshot(a...)
    quote
        dictopen("/tmp/snapshot.dictfile") do df
            for i = 1:length(a)
                #@show a[1] $a[i]
                df[string(a[i])] = a[i]
            end
        end
    end
end

loadsnapshot() = DictFile("/tmp/snapshot.dictfile","r")
 
