function MapFeature(X1,X2,d)

    m=size(X1,1)
    n=((d+d^2)/2)+d

    p=zeros(m,round(Int,n))
    count=0
    for i =1:d
        for j=0:i
            count=count+1
            p[:,count]=(X1.^(i-j)).*(X2.^j)
        end
    end

    return p
end


function PolyFeature(X,d)

    m=size(X,1)
    p=zeros(m,d)

    for i=1:d
        p[:,i]=X.^i
    end

    return p

end
