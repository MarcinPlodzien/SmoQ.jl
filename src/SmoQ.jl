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

## Quantum channels

include("cpu/cpuHamiltonianBuilder.jl")
include("cpu/cpuQuantumChannelGates.jl")
include("cpu/cpuQuantumChannelKrausOperators.jl")
include("cpu/cpuQuantumChannelLindbladianEvolutionMonteCarloWaveFunction.jl")
include("cpu/cpuQuantumChannelRandomUnitaries.jl")
include("cpu/cpuQuantumChannelUnitaryEvolutionExact.jl")
include("cpu/cpuQuantumChannelUnitaryEvolutionTrotter.jl")
include("cpu/cpuQuantumChannelUnitaryEvolutionChebyshev.jl")
# └── CPUHamiltonianBuilder
include("cpu/cpuQuantumChannelLindbladianEvolution.jl")
# ├── CPUHamiltonianBuilder
# ├── CPUQuantumChannelUnitaryEvolutionTrotter
# ├── CPUQuantumChannelUnitaryEvolutionChebyshev
# └── CPUQuantumChannelUnitaryEvolutionExact
include("cpu/cpuQuantumChannelLindbladianEvolutionDensityMatrix.jl")

## Quantum states

include("cpu/cpuQuantumStateMeasurements.jl")
include("cpu/cpuQuantumStatePrintingHelpers.jl")
include("cpu/cpuQuantumStateCharacteristic.jl")
include("cpu/cpuQuantumStateTomography.jl")
include("cpu/cpuQuantumStateLanczos.jl")
include("cpu/cpuQuantumStatePartialTrace.jl")
include("cpu/cpuQuantumStatePreparation.jl")
# └── CPUQuantumStatePartialTrace
include("cpu/cpuQuantumStateObservables.jl")
# ├── CPUQuantumStatePreparation
# └── CPUQuantumStatePartialTrace
include("cpu/cpuQuantumFisherInformation.jl")
# ├── CPUQuantumChannelGates
# └── CPUQuantumStatePartialTrace
include("cpu/cpuQuantumStateManyBodyBellCorrelator.jl")
# └── CPUQuantumChannelGates
include("cpu/cpuClassicalShadows.jl")
# ├── CPUQuantumStateMeasurements
# └── CPUQuantumChannelGates

## Stabilizer Rényi entropy

include("cpu/cpuStabilizerRenyiEntropyBruteForce.jl")
include("cpu/cpuStabilizerRenyiEntropyFastWalshHadamardTransform.jl")

## QRC Simulation

include("cpu/cpuQRCstep.jl")
# ├── CPUQuantumStatePreparation
# ├── CPUQuantumChannelGates
# └── CPUQuantumStatePartialTrace
include("cpu/cpuQRCSimulationMonteCarloWaveFunction.jl")
# ├── CPUQuantumChannelUnitaryEvolutionTrotter
# ├── CPUQuantumStateObservables
# └── CPUQRCstep
include("cpu/cpuQRCSimulationDensityMatrix.jl")
# ├── CPUQuantumStateObservables
# ├── CPUQuantumStateCharacteristic
# ├── CPUQuantumChannelUnitaryEvolutionTrotter
# └── CPUQRCstep

## Variational quantum circuits

include("cpu/cpuVariationalQuantumCircuitOptimizers.jl")
include("cpu/cpuVariationalQuantumCircuitGradients.jl")
include("cpu/cpuVariationalQuantumCircuitCostFunctions.jl")
include("cpu/cpuVariationalQuantumCircuitBuilder.jl")
# └── CPUQuantumChannelGates
include("cpu/cpuVQCEnzymeWrapper.jl")
# └── CPUQuantumChannelGates
include("cpu/cpuVariationalQuantumCircuitExecutor.jl")
# ├── CPUVariationalQuantumCircuitBuilder
# ├── CPUQuantumChannelGates
# ├── CPUQuantumStateMeasurements
# ├── CPUQuantumStatePartialTrace
# └── CPUQuantumChannelKraus

# Re-export commonly used functions for external access
using .CPUQuantumStatePreparation
using .CPUQuantumStatePartialTrace
using .CPUQuantumChannelGates
using .CPUHamiltonianBuilder

# Helper utilities
include("helpers/TaskEvaluationHelper.jl")
include("helpers/DataManagementHelper.jl")
include("helpers/PlottingHelper.jl")
# └── DataManagementHelper

end
