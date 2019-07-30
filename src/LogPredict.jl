function LogPredict(θ;μ=0,σ=1,ξ=0.5)
       P=function (X)

      m=size(X,1)
      s=zeros(m)
      X=(X.-μ)./σ
      X=[ones(m) X]
      p=sigmoid(X*θ)
      s[p.>=ξ] .=1
        return s
    end

        return P
end
