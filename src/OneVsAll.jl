function OneVsAll(X,y;λs=0,Norms=false)
   m=size(X,1)
   n=size(X,2)
   NLabels=size(unique(y),1)

    Θ=zeros(NLabels,n+1)

    for i=1:NLabels
       Cost, grad, avg, st=LogCostGradient(X,(y.==i).*1.0,λ=λs, Norm=Norms)
       θi=zeros(n+1)
       r=optimize(Cost,grad,θi,LBFGS(),inplace = false)
        Θ[i,:]=r.minimizer

    end


    return Θ
end
