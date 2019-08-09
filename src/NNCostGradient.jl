function NNCostGradient(X,y,LayerStruct;λ=0)
    m=size(X,1)                  #Number of training samples
    n=size(X,2)                  #Number of Features
    k=size(LayerStruct,1)-1           #Number of additional layers ( It does not include the first layer)

    ΘNparams=ΘNparams=Array{Int64,1}(undef,k)

    for i=1:k
       ΘNparams[i]=(LayerStruct[i]+1)*LayerStruct[i+1]
    end


    CostFunction=function (θ)
                    Θ=Array{Array{Float64,2},1}(undef, k)   #Initialize the shape of the Θ Array of Arrays
                    a=Array{Array{Float64,2},1}(undef, k+1)
                    z=Array{Array{Float64,2},1}(undef, k)
                    a[1]=[ones(m) X]
                    r=zeros(k)
                    c=0
                    for i=1:k
                        Θ[i]=reshape(θ[c+1:c+ΘNparams[i]],(LayerStruct[i+1],LayerStruct[i]+1))
                        c=ΘNparams[i]
                        z[i]=a[i]*Θ[i]'
                        a[i+1]=Sigmoid(z[i])
                        a[i+1]=[ones(m) a[i+1]]
                        r[i]=sum(Θ[i].^2)
                    end

                    labels=l=Vector(1:LayerStruct[end])'
                    Y=(y.==labels).*1.0
        J=(-1/m)*sum(Y.*log.(a[end][:,2:end]).+(1 .-Y).*log.(1 .-a[end][:,2:end]))+((0.5*λ/m).*sum(r))
                    return J
    end

        Gradient = function (θ)
                            Θ=Array{Array{Float64,2},1}(undef, k)   #Initialize the shape of the Θ Array of Arrays
                    a=Array{Array{Float64,2},1}(undef, k+1)
                    z=Array{Array{Float64,2},1}(undef, k)
                    a[1]=[ones(m) X]
                    r=zeros(k)
                    c=0
                    for i=1:k
                        Θ[i]=reshape(θ[c+1:c+ΘNparams[i]],(LayerStruct[i+1],LayerStruct[i]+1))
                        c=ΘNparams[i]
                        z[i]=a[i]*Θ[i]'
                        a[i+1]=Sigmoid(z[i])
                        a[i+1]=[ones(m) a[i+1]]

                    end
                    δ=Array{Array{Float64,2},1}(undef, k)
                    ∇=Array{Array{Float64,2},1}(undef, k)
                    labels=l=Vector(1:LayerStruct[end])'
                    Y=(y.==labels).*1.0
                    δ[end]=a[end][:,2:end].-Y
                    ∇[end]=((δ[end]'*a[end-1])./m).+(λ./m).*[zeros(size(Θ[end],1)) Θ[end][:,2:end]]
                    for i=k-1:-1:1
                    δ[i]=(δ[i+1]*Θ[i+1]).*SigmoidGradient([ones(m) z[i]])
                    ∇[i]=((δ[i][:,2:end]'*a[i])./m).+(λ./m).*[zeros(size(Θ[i],1)) Θ[i][:,2:end]]
                     end

                    ∇f=∇[1][:]
                    for i=2:k

                        ∇f=vcat(∇f,∇[i][:])

                    end

        return ∇f

                end
    return CostFunction, Gradient,ΘNparams
end
