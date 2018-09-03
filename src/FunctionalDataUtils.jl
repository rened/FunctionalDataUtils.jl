module FunctionalDataUtils

using Distributed, Statistics

LOGFILE = ""
LOGTOFILE = false

function __init__()
    global LOGFILE
    LOGFILE = joinpath(pwd(), "julia.log")
end

FDU = FunctionalDataUtils
export FDU

import FunctionalData.apply
using FunctionalData#, Colors
using SHA

include("computing.jl")
include("numerical.jl")
include("computervision.jl")
include("graphics.jl")
include("machinelearning.jl")
# include("matlab.jl")
include("output.jl")
include("utils.jl")
include("system.jl")
include("sampler.jl")

end # module
