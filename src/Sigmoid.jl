function Sigmoid(x)
   g=1 ./ (1 .+exp.(-x))
    return g
end

function SigmoidGradient(x)

    g=Sigmoid(x).*(1 .- Sigmoid(x))
    return g

end
