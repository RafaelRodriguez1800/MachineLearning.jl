function DataSplit(X,Y;p=[0.6,0.2,0.2])
   m=size(X,1)
   TrNumber=round(Int,m*p[1])
   CvNumber=round(Int,m*p[2])
   TsNumber=round(Int,m*p[3])

    #Ramdomly sampled the X inputs
    r = sample(1:m, m, replace = false)
    Xr=X[r,:]
    Yr=Y[r,:]

    #Get X and Y train
    Xtr=Xr[1:TrNumber,:]
    Ytr=Yr[1:TrNumber]

    #Get X and Y CrossValidation
    Xcv=Xr[TrNumber+1:TrNumber+CvNumber,:]
    Ycv=Yr[TrNumber+1:TrNumber+CvNumber]

    #Get X and Y Test
    Xts=Xr[TrNumber+CvNumber+1:end,:]
    Yts=Yr[TrNumber+CvNumber+1:end]

    return Xtr, Ytr, Xcv, Ycv, Xts, Yts

end
