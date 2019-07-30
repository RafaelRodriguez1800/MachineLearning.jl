module MachineLearning

using Optim
using LinearAlgebra


export CostGradient, LogCostGradient, LogPredict, Predict, OneVsAll,
        OneVsAllPredict, MapFeature, PolyFeature,Sigmoid, Kmeans, FindCentroids,
            ComputeCentroids

include("CostGradient.jl")
include("LogCostGradient.jl")
include("LogPredict.jl")
include("OneVsAll.jl")
include("OneVsAllPredict.jl")
include("PolyFeatures.jl")
include("Predict.jl")
include("Sigmoid.jl")
include("Kmeans.jl")
include("FindCentroids.jl")
include("ComputeCentroids.jl")











end # module
