function LogCostGradient(X::Array,y::Array;λ=0, Norm=false)

   #Size of the X Parameters
    m=size(X,1)
    n=size(X,2)
    #if Normalize

    if Norm==true
       μ=mean(X,dims=1)
       σ=std(X,dims=1)
       X=(X.-μ)./σ
    else
        μ=zeros(1,n)
        σ=ones(1,n)
    end


    #Add vector of ones to X parameters
    X=[ones(m) X]

    CostFunction=function (θ)
                    J=((1/m).*((-1 .*y.*log.(Sigmoid(X*θ)).-(1 .-y).*log.(1 .-Sigmoid(X*θ)))'*ones(m))).+((0.5*λ/m).*(θ[2:end]'*θ[2:end]))
                    return J
                end
    θr=zeros(n+1)
    Gradient=function (θ)
             θr.=θ
             θr[1]=0
             g=(1/m).*(X'*(Sigmoid(X*θ).-y)) .+ ((λ./m).*θr)
        return g
            end

    return CostFunction, Gradient, μ, σ
end
