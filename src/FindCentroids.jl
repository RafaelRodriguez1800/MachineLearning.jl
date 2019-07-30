function FindCentroids(X,C)
   m=size(X,1)
   n=size(X,2)
   k=size(C,1)

   dis=zeros(m,k)
   idx=zeros(m)
    for i=1:k
       dis[:,i]=sqrt.(sum((X.-C[i,:]').^2,dims=2))
    end
    for i=1:m
        idx[i]=argmin(dis[i,:])
    end

    return idx


end
