using DictFiles

export snapshot, loadsnapshot

# macro snapshot(a...)
#     quote
        # dictopen("/tmp/snapshot.dictfile") do df
        #     for i = 1:length(a)
        #         #@show a[1] $a[i]
        #         df[string(a[i])] = [i]
        #     end
        # end
    # end
# end

function snapshot(a...)
    dictopen("/tmp/snapshot.dictfile","w") do df
        for i = 1:2:length(a)
            df[a[i]] = a[i+1]
        end
    end
end

loadsnapshot(a...) = dictopen("/tmp/snapshot.dictfile","r") do df
    if length(a) == 0
        df[]
    else
        [df[x] for x in a]
    end
end
 
