function ComputeCentroids(X,idx)
    n=size(X,2)
    k=size(unique(idx),1)
    C=zeros(k,n)

    for i=1:k
        C[i,:]=mean(X[idx.==i,:],dims=1)
    end

    return C
end
