module SmoQ

using BenchmarkTools
using BlackBoxOptim
using CUDA
using Enzyme
using ExponentialUtilities
using JLD2
using Krylov
using KrylovKit
using LinearAlgebra
using LinearOperators
using LsqFit
using Optimisers
using OrdinaryDiffEq
using Plots
using ProgressMeter
using SparseArrays
using SpecialFunctions
using Statistics
using Zygote

# Include all CPU submodules
include("cpu/cpuClassicalShadows.jl")
include("cpu/cpuQuantumStatePreparation.jl")
include("cpu/cpuQuantumStatePartialTrace.jl")
include("cpu/cpuQuantumChannelGates.jl")
include("cpu/cpuHamiltonianBuilder.jl")
include("cpu/cpuQuantumChannelUnitaryEvolutionTrotter.jl")
include("cpu/cpuQuantumChannelUnitaryEvolutionChebyshev.jl")
include("cpu/cpuQuantumChannelUnitaryEvolutionExact.jl")
include("cpu/cpuQRCSimulationDensityMatrix.jl")
include("cpu/cpuQRCSimulationMonteCarloWaveFunction.jl")
include("cpu/cpuQRCstep.jl")
include("cpu/cpuQuantumChannelKrausOperators.jl")
include("cpu/cpuQuantumChannelLindbladianEvolution.jl")
include("cpu/cpuQuantumChannelLindbladianEvolutionDensityMatrix.jl")
include("cpu/cpuQuantumChannelLindbladianEvolutionMonteCarloWaveFunction.jl")
include("cpu/cpuQuantumChannelRandomUnitaries.jl")
include("cpu/cpuQuantumFisherInformation.jl")
include("cpu/cpuQuantumStateCharacteristic.jl")
include("cpu/cpuQuantumStateLanczos.jl")
include("cpu/cpuQuantumStateManyBodyBellCorrelator.jl")
include("cpu/cpuQuantumStateMeasurements.jl")
include("cpu/cpuQuantumStateObservables.jl")
include("cpu/cpuQuantumStatePrintingHelpers.jl")
include("cpu/cpuQuantumStateTomography.jl")
include("cpu/cpuStabilizerRenyiEntropyBruteForce.jl")
include("cpu/cpuStabilizerRenyiEntropyFastWalshHadamardTransform.jl")
include("cpu/cpuVQCEnzymeWrapper.jl")
include("cpu/cpuVariationalQuantumCircuitBuilder.jl")
include("cpu/cpuVariationalQuantumCircuitCostFunctions.jl")
include("cpu/cpuVariationalQuantumCircuitExecutor.jl")
include("cpu/cpuVariationalQuantumCircuitGradients.jl")
include("cpu/cpuVariationalQuantumCircuitOptimizers.jl")

# Re-export commonly used functions for external access
using .CPUQuantumStatePreparation
using .CPUQuantumStatePartialTrace  
using .CPUQuantumChannelGates
using .CPUHamiltonianBuilder

# Helper utilities
include("helpers/DataManagementHelper.jl")
include("helpers/PlottingHelper.jl")
include("helpers/TaskEvaluationHelper.jl")

end
