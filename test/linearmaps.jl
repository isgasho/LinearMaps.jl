using Test, LinearMaps, LinearAlgebra, SparseArrays, BenchmarkTools

@testset "basic functionality" begin
    A = 2 * rand(ComplexF64, (20, 10)) .- 1
    v = rand(ComplexF64, 10)
    u = rand(ComplexF64, 20, 1)
    V = rand(ComplexF64, 10, 3)
    W = rand(ComplexF64, 20, 3)
    α = rand()
    β = rand()
    M = @inferred LinearMap(A)
    N = @inferred LinearMap(M)

    @testset "LinearMaps.jl" begin
        @test eltype(M) == eltype(A)
        @test size(M) == size(A)
        @test size(N) == size(A)
        @test !isreal(M)
        @test ndims(M) == 2
        @test_throws ErrorException size(M, 3)
        @test length(M) == length(A)
    end

    @testset "dimension checking" begin
        w = vec(u)
        @test_throws DimensionMismatch M * similar(v, length(v) + 1)
        @test_throws DimensionMismatch mul!(similar(w, length(w) + 1), M, v)
        @test_throws DimensionMismatch similar(w, length(w) + 1)' * M
        @test_throws DimensionMismatch mul!(copy(v)', similar(w, length(w) + 1)', M)
        @test_throws DimensionMismatch mul!(similar(W, size(W).+(0,1)), M, V)
        @test_throws DimensionMismatch mul!(copy(W), M, similar(V, size(V).+(0,1)))
    end
end

# new type
struct SimpleFunctionMap <: LinearMap{Float64}
    f::Function
    N::Int
end
struct SimpleComplexFunctionMap <: LinearMap{Complex{Float64}}
    f::Function
    N::Int
end
Base.size(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}) = (A.N, A.N)
Base.:(*)(A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, v::AbstractVector) = A.f(v)
LinearAlgebra.mul!(y::AbstractVector, A::Union{SimpleFunctionMap,SimpleComplexFunctionMap}, x::AbstractVector) = copyto!(y, *(A, x))

@testset "new LinearMap type" begin
    F = SimpleFunctionMap(cumsum, 10)
    FC = SimpleComplexFunctionMap(cumsum, 10)
    @test @inferred ndims(F) == 2
    @test @inferred size(F, 1) == 10
    @test @inferred length(F) == 100
    @test @inferred !issymmetric(F)
    @test @inferred !ishermitian(F)
    @test @inferred !ishermitian(FC)
    @test @inferred !isposdef(F)
    @test occursin("10×10 SimpleFunctionMap{Float64}", sprint((t, s) -> show(t, "text/plain", s), F))
    @test occursin("10×10 SimpleComplexFunctionMap{Complex{Float64}}", sprint((t, s) -> show(t, "text/plain", s), FC))
    α = rand(ComplexF64); β = rand(ComplexF64)
    v = rand(ComplexF64, 10); V = rand(ComplexF64, 10, 3)
    w = rand(ComplexF64, 10); W = rand(ComplexF64, 10, 3)
    @test mul!(w, F, v) === w == F * v
    @test_throws ErrorException F' * v
    @test_throws ErrorException transpose(F) * v
    @test_throws ErrorException mul!(w, adjoint(FC), v)
    @test_throws ErrorException mul!(w, transpose(F), v)
    FM = convert(AbstractMatrix, F)
    L = LowerTriangular(ones(10, 10))
    @test FM == L
    @test F * v ≈ L * v
    # generic 5-arg mul! and matrix-mul!
    @test mul!(copy(w), F, v, α, β) ≈ L*v*α + w*β
    @test mul!(copy(W), F, V) ≈ L*V
    @test mul!(copy(W), F, V, α, β) ≈ L*V*α + W*β

    Fs = sparse(F)
    @test Fs == L
    @test Fs isa SparseMatrixCSC
end
