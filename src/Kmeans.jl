function Kmeans(X,k;MaxIter=20)
    m=size(X,1)
    n=size(X,2)

    #Initial Centroids
    C=X[rand(1:m,k),:]
    Idx=zeros(m,MaxIter)
    for i=1:MaxIter
       Idx[:,i]=FindCentroids(X,C)
        C=ComputeCentroids(X,Idx[:,i])
    end

    return Idx[:,end]
end
