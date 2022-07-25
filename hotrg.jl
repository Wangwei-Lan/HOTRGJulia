
#        include files
#include("../../2DClassical/partition.jl")
include("HOTRGfunc.jl")






#---- parameter
nLayer=24
BetaExact =1/2*log(1+sqrt(2))
#T = 2.5#1/Beta
#Beta = 1/T#0.9994*BetaExact
E0=-0.0
Dlink = 2



#


#----- Debug paramenters
FEtrg = Array{Float64}(undef,0)
NumSite = Array{Int64}(undef,0)
FreeEnergyExact = Array{Float64}(undef,0)
Amatrix = []
Slmatrix = []
Srmatrix = []

S1matrix = []
S2matrix = []
S3matrix = []
S4matrix = []

#for chimax in 10:10:50
#for T in 2.1:0.05:2.5
T = 2/log(2+sqrt(2))
Beta = 1/T
#------------------------initial A Tensor----------------------------
Q = Float64[exp(Beta) exp(-Beta);
            exp(-Beta) exp(Beta)]
g = zeros(Dlink,Dlink,Dlink,Dlink)
g[1,1,1,1] = g[2,2,2,2] = 1
gp = zeros(Dlink,Dlink,Dlink,Dlink)
gp[1,1,1,1] = -1;gp[2,2,2,2] = 1
@tensor A[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*g[1,2,3,4]
@tensor Asx[-1,-2,-3,-4] := sqrt(Q)[-1,1]*sqrt(Q)[-2,2]*sqrt(Q)[-3,3]*sqrt(Q)[-4,4]*gp[1,2,3,4]
#


#------   contract 4 tensor into 1 tensor, this step can be deleted
@tensor A0[-1,-2,-3,-4,-5,-6,-7,-8] := A[-2,-3,2,1]*A[2,-4,-6,3]*A[4,3,-5,-8]*A[-1,1,4,-7]
A = reshape(A0,4,4,4,4)
feexact = ComputeFreeEnergy(Beta)
    chimax = 10
   # A = A0
    #-------   Exact value of Free Energy
    #-------   NormFactor stores the normalization coefficient that is used to prevent divergence in tensors
    NormFactor = Array{Float64}(undef,0)
    append!(NormFactor,maximum(A))
    global A = A./maximum(A)
for i in 1:nLayer
    global A;
    print("-----------------------This is layer i: ", i,"----------------------------\n")
    @time begin
    #---- bond dimension
    Alink1 = size(A)[1]  #try metaprogramming
    Alink2 = size(A)[2]
    Alink3 = size(A)[3]
    Alink4 = size(A)[4]
    #--- Left and Right are two reshapes that will svd into S1-4
    println(size(A))
    @time Projector = ComputeProjector(A,chimax)

    @time @tensor A[-1,-2,-3,-4] := Projector[1,2,-2]*Projector[4,5,-4]*A[-1,1,3,4]*A[3,2,-3,5]


    Projector = ComputeProjector(permutedims(A,[2,3,4,1]),chimax)

    @tensor A[-1,-2,-3,-4] := Projector[1,3,-3]*Projector[4,5,-1]*A[4,-2,1,2]*A[5,2,3,-4]


    append!(NormFactor,[maximum(A)])
    A = A./maximum(A)
    #
    Ns = 4^(i+1)
    append!(NumSite,Ns)
    @tensor Z[] := A[1,2,1,2]
    FE= (log(Z[1])+    sum([4^(i-j+1) for j in 1:i+1].*log.(NormFactor)) )/Ns
    append!(FEtrg,[FE])
    #
    end
end

Ns = 4^(nLayer+1)
append!(NumSite,Ns)
@tensor Z[] := A[1,2,1,2]
FE= (log(Z[1])+    sum([4^(nLayer-j+1) for j in 1:nLayer+1].*log.(NormFactor)) )/Ns
append!(FEtrg,[FE])
append!(FreeEnergyExact,[feexact])

#end
#end
