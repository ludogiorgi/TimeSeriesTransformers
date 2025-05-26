# ===================================================================
# Lorenz Time Series Prediction with Transformer
# ===================================================================
#
# This example demonstrates how to use TimeSeriesTransformers to:
# 1. Generate Lorenz 63 chaotic time series data
# 2. Train a transformer model to predict the next value in the sequence
# 3. Generate long time series predictions
# 4. Visualize results and analyze performance
#exit
# The example showcases the full workflow from data generation to model
# evaluation and time series generation.
using Revise

using TimeSeriesTransformers
using Flux
using Random
using Plots
using Statistics

# Set seed for reproducibility
Random.seed!(42)

println("=== Lorenz Time Series Transformer Example ===")
println("Julia threads: $(Threads.nthreads())")

# ===================================================================
# 1. Data Generation and Preprocessing
# ===================================================================

println("\n1. Generating Lorenz 63 time series data...")

# Generate chaotic time series from Lorenz 63 system
# The Lorenz system is a classic example of deterministic chaos
# Reduce data size to avoid concatenation overflow
data = generate_lorenz63_data(20000, tspan=(0.0, 2000.0))

# Extract the y-variable for time series prediction
y_data = data[:, 2]

println("Generated $(length(y_data)) data points")
println("Data range: [$(round(minimum(y_data), digits=3)), $(round(maximum(y_data), digits=3))]")

# Create discrete clusters from continuous data
num_clusters = 10
println("\n2. Creating $(num_clusters) clusters from continuous data...")

processor = TimeSeriesProcessor(y_data, num_clusters)
println("Cluster centers: $(round.(processor.cluster_centers, digits=3))")

# ===================================================================
# 2. Model Configuration
# ===================================================================

println("\n3. Configuring transformer model...")

# Model hyperparameters - adjusted for stability
sequence_length = 64    # Reduced sequence length
d_model = 32           # Model dimension
num_heads = 8          # Number of attention heads
num_layers = 1         # Reduced number of transformer layers
dropout_rate = 0.1     # Dropout rate for regularization

# Validate configuration
head_dim = d_model ÷ num_heads
if d_model % num_heads != 0
    error("d_model ($d_model) must be divisible by num_heads ($num_heads)")
end

println("Model configuration:")
println("  - Sequence length: $sequence_length")
println("  - Model dimension: $d_model")
println("  - Attention heads: $num_heads (head dimension: $head_dim)")
println("  - Transformer layers: $num_layers")
println("  - Dropout rate: $dropout_rate")

# Create the transformer model
model = TransformerModel(
    num_clusters = num_clusters,
    latent_dim = num_clusters,
    d_model = d_model,
    num_heads = num_heads,
    num_layers = num_layers,
    dropout_rate = dropout_rate
)

# Count parameters
param_count = sum(length, Flux.params(model))
println("Total trainable parameters: $param_count")

# ===================================================================
# 3. Training and Evaluation with Checkpointing
# ===================================================================

println("\n4. Training transformer model with checkpointing...")

# Train model with adjusted parameters to avoid memory issues
model, train_losses, val_losses = train_and_evaluate!(
    model, 
    processor, 
    sequence_length;
    epochs = 100,
    batch_size = 16,      # Increased batch size to process fewer total batches
    learning_rate = 1f-3,
    early_stopping_patience = 10,
    verbose = true,
    display_plots = true,
    n_batches_per_epoch = 200,  # Increased to handle larger dataset efficiently
    checkpoint_dir = "checkpoints",
    checkpoint_frequency = 5,
    resume_from_checkpoint = false,
    save_final_model = true,
    model_filename = "my_lorenz_transformer",
    analysis_callback_frequency = 5,
    processor = processor,
    y_data = y_data
)

println("Training completed successfully!")

##

