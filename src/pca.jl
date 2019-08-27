function pca(X::Array, k::Int)

    m=size(X)
    n=size(X)

    μ=mean(X,dims=1)
    σ=std(X,dims=1)
    Xn=(X.-μ)./σ

    CovM=(1/m).*(Xn'*Xn)
    U,S,V=svd(CovM)
    Ur=U[:,1:k]
    Z=Xn*Ur

    return Z
end
