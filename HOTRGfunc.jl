"""
    ConstructLocalTensor(Beta::Float64)

Construct the local tensor A_o in square lattice for classical Ising Model
Return A_o; A_osx, which for spin down, have a extra negative sign
"""
function ConstructLocalTensor(Beta::Float64) ::Tuple{Array{Float64},Array{Float64}}#::Tuple{::Array{Float64},::Array{Float64}}
    #---- Q locate in between two spins
    Q = Float64[exp(Beta) exp(-Beta);
                exp(-Beta) exp(Beta)]
    #---- g locate on site, indicating that spin are the same
    g = zeros(2,2,2,2)
    g[1,1,1,1] = g[2,2,2,2] = 1
    #---- gp is also locate on site, but with negative sign for down spin
    gp = zeros(2,2,2,2)
    gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
    @tensor A[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
    @tensor Asx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*gp[1,2,3,4]
    return A,Asx
end



"""

Construct Projector for Higher Order TRG
"""
function ComputeProjector(A::Array{Float64},chimax::Int64)

    chiA = size(A,2);
    @tensor TEMP[-1,-2,-3,-4] := A[1,-1,5,2]*A[5,-2,3,4]*A[1,-3,6,2]*A[6,-4,3,4]
    TEMP = reshape(TEMP,chiA^2,chiA^2)

    F = svd((TEMP+TEMP')/2)
    Projector = reshape(F.U,chiA,chiA,chiA^2)
    if chiA^2 > chimax
        Projector = Projector[1:chiA,1:chiA,1:chimax]
    end
    return Projector

      
end


"""
    CompBinder

Compute higher order moment in HOTRG scheme
"""
function CompHigherOrderMoment(A,S1,S2,S3,S4,PL,PR)

    @tensor AS1[:] := A[-1,1,3,4]*S1[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S1A[:] := S1[-1,1,3,4]*A[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]

    @tensor AS2[:] := A[-1,1,3,4]*S2[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S2A[:] := S2[-1,1,3,4]*A[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S1S1[:] := S1[-1,1,3,4]*S1[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]

    @tensor AS3[:] :=A[-1,1,3,4]*S3[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S3A[:] := S3[-1,1,3,4]*A[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S1S2[:] := S1[-1,1,3,4]*S2[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S2S1[:] := S2[-1,1,3,4]*S1[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]


    @tensor AS4[:] :=A[-1,1,3,4]*S4[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S4A[:] := S4[-1,1,3,4]*A[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S1S3[:] := S1[-1,1,3,4]*S3[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S3S1[:] := S3[-1,1,3,4]*S1[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S2S2[:] := S2[-1,1,3,4]*S2[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]

    S1 = 0.5*(AS1+S1A);S2 = 0.25*(AS2+S2A+2*S1S1)
    S3 = 0.125*(AS3+S3A+3*S2S1+3*S1S2)
    S4 = 1/16*(AS4+S4A+4*S3S1+4*S1S3+6*S2S2)
    return S1,S2,S3,S4
end



function compute_mag(A,S1,PL,PR)

    @tensor AS1[:] := A[-1,1,3,4]*S1[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    @tensor S1A[:] := S1[-1,1,3,4]*A[3,2,-3,5]*PL[4,5,-4]*PR[1,2,-2]
    S1 = 0.5*(AS1+S1A);

    return S1
end