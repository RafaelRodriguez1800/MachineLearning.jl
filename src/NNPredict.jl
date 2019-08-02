function NNPredict(θ,LayerStruct)
 k=size(LayerStruct,1)-1           #Number of additional layers ( It does not include the first layer)

    ΘNparams=ΘNparams=Array{Int64,1}(undef,k)

    for i=1:k
       ΘNparams[i]=(LayerStruct[i]+1)*LayerStruct[i+1]
    end
    p= function (X)
        m=size(X,1)
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

        s=zeros(m)
        for i=1:m
            s[i]=argmax(a[end][i,2:end])
        end
        return s
    end
        return p
end
