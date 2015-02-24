export exitwithteststatus

function exitwithteststatus()
    s = FactCheck.getstats()
    if s["nNonSuccessful"] == 0
        print("   PASSED!")
    else
        print("   FAILED: $(s["nFailures"]) failures and $(s["nErrors"]) errors")
    end
    println("   ( $(s["nNonSuccessful"]+s["nSuccesses"]) tests for runtests.jl $(join(ARGS, " ")))")
    exit(s["nNonSuccessful"])
end

