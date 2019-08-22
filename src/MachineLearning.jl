module MachineLearning

using Optim
using LinearAlgebra
using Distributions


export CostGradient, LogCostGradient, LogPredict, Predict, OneVsAll,
        OneVsAllPredict, MapFeature, PolyFeature,Sigmoid, Kmeans, FindCentroids,
            ComputeCentroids, SigmoidGradient, NNPredict, NNCostGradient, MeshGrid, DataSplit

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
include("NNPredict.jl")
include("NNCostGradient.jl")
include("MeshGrid.jl")
include("DataSplit.jl")












end # module
