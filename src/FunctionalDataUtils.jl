__precompile__()

module FunctionalDataUtils

LOGFILE = ""
LOGTOFILE = false

function __init__()
    global LOGFILE
    LOGFILE = joinpath(pwd(), "julia.log")
end

FDU = FunctionalDataUtils
export FDU

using Reexport
@reexport using FunctionalData, Colors
using SHA
import FactCheck

include("computing.jl")
include("numerical.jl")
include("computervision.jl")
include("fmIO.jl")
include("graphics.jl")
include("machinelearning.jl")
# include("matlab.jl")
include("numerical.jl")
include("output.jl")
include("utils.jl")
include("system.jl")
include("sampler.jl")

end # module
