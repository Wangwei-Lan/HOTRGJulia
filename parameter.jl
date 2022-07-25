mutable struct TRGparam
        step::Int64
        temperature::Float64
        beta::Float64
        chimax::Int64
        dlink::Int64      # physical link bond dimension for A_o
        A::Array{Float64}
        SingularArray::Array{Array{Float64}}
        NormArray::Array{Float64}
        SiteArray::Array{Float64}
        FreeEnergyArray::Array{Float64}
        function TRGparam(step::Int64,temperature::Float64,chimax::Int64)
            dlink = 2
            Beta = 1/temperature
            A,Asx = ConstructLocalTensor(Beta)
            @tensor A0[-1,-2,-3,-4,-5,-6,-7,-8] := A[-2,-3,2,1]*A[2,-4,-6,3]*A[4,3,-5,-8]*A[-1,1,4,-7]
            A = reshape(A0,4,4,4,4)
            SingularArray=Array{Array{Float64}}(undef,step)
            NormArray= Array{Float64}(undef,step+1)
            SiteArray=Array{Float64}(undef,step)
            FreeEnergyArray= Array{Float64}(undef,step)
            new(step,temperature,Beta,chimax,dlink,A,SingularArray,NormArray,SiteArray,FreeEnergyArray)
        end
end
