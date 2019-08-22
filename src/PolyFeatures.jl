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
    n=size(X,2)
    p=zeros(m,d*n)
c=0
for j=1:n
    for i=1:d
        c=c+1
        p[:,c]=X[:,j].^i
    end
end
    return p

end
