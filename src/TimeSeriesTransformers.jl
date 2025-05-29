"""
# TimeSeriesTransformers.jl

A Julia package for time series prediction using transformer neural networks.

## Features
- Complete transformer architecture implementation
- Time series discretization and preprocessing  
- Chaotic system data generation (Lorenz 63)
- Comprehensive training utilities with early stopping
- Visualization and analysis tools

## Usage
```julia
using TimeSeriesTransformers

# Generate data and create model
data = generate_lorenz63_data(1000)
processor = TimeSeriesProcessor(data[:, 2], 10)
model = TransformerModel(num_clusters=10, latent_dim=10, d_model=64, num_heads=8, num_layers=2)

# Train and evaluate
train_and_evaluate!(model, processor, 16)
```
"""
module TimeSeriesTransformers

using Flux
using Clustering
using Statistics
using LinearAlgebra
using Random
using DifferentialEquations
using JLD2
using NNlib
using KernelDensity

# Import and export main components
include("transformer.jl")
include("time_series_processor.jl")
include("data_generator.jl")
include("model_io.jl")
include("training.jl")
include("prediction_utils.jl")
include("analysis_callback.jl")

# Export main functions and types
export TimeSeriesProcessor, TransformerModel
export train_transformer!, train_and_evaluate!
export generate_lorenz63_data
export get_sequence_dataset

# Export new autoregressive functions
export prepare_autoregressive_targets, prepare_autoregressive_batch
export create_autoregressive_batch_tensor
export predict_next_token, generate_sequence, predict_sequence_continuation
export sample_from_distribution

# Export utility functions from prediction_utils
export generate, ensemble_predict, compute_validation_accuracy

# Export transformer components
export TransformerModel, MultiHeadAttention, FeedForward
export TransformerEncoderLayer, TransformerEncoder, PositionalEncoding
export transformer_loss, stable_softmax

# Export model methods
export train!, predict_probabilities

# Export training utilities
export train_with_validation!, train_and_evaluate!

# Export utility functions
export save_model, load_model, set_threading

# Export analysis callback and evaluation functions
export comprehensive_analysis_callback, autocorrelation, initialize_analysis_params!
export evaluate_autoregressive_model

end # module
