# ===================================================================
# Lorenz Time Series Prediction with Transformer
# ===================================================================
#
# This example demonstrates how to use TimeSeriesTransformers to:
# 1. Generate Lorenz 63 chaotic time series data
# 2. Train an autoregressive transformer model
# 3. Generate long time series predictions using autoregressive generation
# 4. Visualize results and analyze performance
#
# The example showcases the full workflow from data generation to model
# evaluation and autoregressive time series generation.

using Pkg

# Ensure we're using the local development version
Pkg.activate(".")

using TimeSeriesTransformers
using Flux
using Random
using Plots
using Statistics

# Set seed for reproducibility
Random.seed!(42)

println("=== Lorenz Autoregressive Transformer Example ===")
println("Julia threads: $(Threads.nthreads())")

# ===================================================================
# 1. Data Generation and Preprocessing
# ===================================================================

println("\n1. Generating Lorenz 63 time series data...")

# Generate chaotic time series from Lorenz 63 system
# The Lorenz system is a classic example of deterministic chaos
data = generate_lorenz63_data(20000, tspan=(0.0, 2000.0))

# Extract the y-variable for time series prediction
y_data = data[:, 2]

println("Generated $(length(y_data)) data points")
println("Data range: [$(round(minimum(y_data), digits=3)), $(round(maximum(y_data), digits=3))]")

# Create discrete clusters from continuous data
num_clusters = 10
println("\n2. Creating $(num_clusters) clusters from continuous data...")

# Create processor and perform clustering
processor = TimeSeriesTransformers.TimeSeriesProcessor(y_data, num_clusters)

# Process the data to perform clustering
TimeSeriesTransformers.process(processor)

println("Cluster centers: $(round.(processor.cluster_centers, digits=3))")
println("Vocabulary size: $(processor.vocab_size)")
println("Number of sequences: $(length(processor.sequences))")

# ===================================================================
# 2. Model Configuration for Autoregressive Training
# ===================================================================

println("\n3. Configuring autoregressive transformer model...")

# Model hyperparameters - adjusted for autoregressive training
sequence_length = 32    # Shorter sequences for autoregressive training
d_model = 64           # Model dimension
num_heads = 8          # Number of attention heads
num_layers = 2         # Number of transformer layers
dropout_rate = 0.1     # Dropout rate for regularization

# Validate configuration
head_dim = d_model ÷ num_heads
if d_model % num_heads != 0
    error("d_model ($d_model) must be divisible by num_heads ($num_heads)")
end

println("Autoregressive model configuration:")
println("  - Sequence length: $sequence_length")
println("  - Model dimension: $d_model")
println("  - Attention heads: $num_heads (head dimension: $head_dim)")
println("  - Transformer layers: $num_layers")
println("  - Dropout rate: $dropout_rate")
println("  - Vocabulary size: $num_clusters")

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
# 3. Autoregressive Training and Evaluation
# ===================================================================

println("\n4. Training autoregressive transformer model...")

# Train model with autoregressive loss
model, train_losses, val_losses = train_and_evaluate!(
    model, 
    processor, 
    sequence_length;
    epochs = 50,
    batch_size = 32,
    learning_rate = 1f-4,
    early_stopping_patience = 8,
    verbose = true,
    display_plots = false,  # Don't display during training
    n_batches_per_epoch = 100,
    checkpoint_dir = "checkpoints",
    checkpoint_frequency = 2,
    resume_from_checkpoint = false,
    save_final_model = true,
    model_filename = "lorenz_autoregressive_transformer",
    analysis_callback_frequency = 2,
    processor = processor,
    y_data = y_data
)

println("Autoregressive training completed successfully!")

# ===================================================================
# 4. Autoregressive Generation and Evaluation
# ===================================================================

println("\n5. Testing autoregressive generation...")

# Get test data for generation
X_test, y_test = get_sequence_dataset(processor, sequence_length)
test_idx = 1:min(10, length(X_test))

println("Evaluating on $(length(test_idx)) test sequences...")

# Test autoregressive generation
for i in test_idx[1:3]  # Test first 3 sequences
    input_seq = X_test[i]
    true_target = isa(y_test[i], AbstractVector) ? y_test[i] : [y_test[i]]
    
    # Generate continuation
    n_generate = 20
    generated_seq = generate_sequence(model, input_seq, num_clusters, n_generate; temperature=0.8f0)
    
    # Convert to original values
    input_values = [processor.vocab_to_value[token] for token in input_seq]
    generated_values = [processor.vocab_to_value[token] for token in generated_seq[(length(input_seq)+1):end]]
    
    println("\nTest sequence $i:")
    println("  Input length: $(length(input_seq))")
    println("  Generated length: $(length(generated_values))")
    println("  Input range: [$(round(minimum(input_values), digits=3)), $(round(maximum(input_values), digits=3))]")
    println("  Generated range: [$(round(minimum(generated_values), digits=3)), $(round(maximum(generated_values), digits=3))]")
end

# Evaluate model performance
if length(X_test) > 0
    eval_results = evaluate_autoregressive_model(model, processor, X_test, y_test; n_samples=20)
    println("\nModel evaluation:")
    println("  Average loss: $(round(eval_results.loss, digits=6))")
    println("  Token accuracy: $(round(eval_results.accuracy * 100, digits=2))%")
end

# ===================================================================
# 5. Visualization
# ===================================================================

println("\n6. Creating visualizations...")

# Plot training curves
if length(train_losses) > 1
    p1 = plot(train_losses, label="Training Loss", title="Autoregressive Training Progress", 
              xlabel="Epoch", ylabel="Loss", lw=2)
    plot!(p1, val_losses, label="Validation Loss", lw=2)
    display(p1)
end

# Generate and plot a longer sequence
if length(X_test) > 0
    println("\n7. Generating extended time series...")
    
    # Use a test sequence as seed
    seed_seq = X_test[1][1:min(16, length(X_test[1]))]  # Use first part as seed
    n_generate = 100
    
    # Generate with different temperatures
    temperatures = [0.5f0, 0.8f0, 1.2f0]
    
    p2 = plot(title="Autoregressive Generation with Different Temperatures", 
              xlabel="Time Step", ylabel="Value", legend=:topright)
    
    # Convert seed to values and plot
    seed_values = [processor.vocab_to_value[token] for token in seed_seq]
    plot!(p2, 1:length(seed_values), seed_values, label="Seed", lw=3, color=:black)
    
    for (i, temp) in enumerate(temperatures)
        generated_seq = generate_sequence(model, seed_seq, num_clusters, n_generate; temperature=temp)
        generated_values = [processor.vocab_to_value[token] for token in generated_seq[(length(seed_seq)+1):end]]
        
        start_idx = length(seed_values) + 1
        end_idx = start_idx + length(generated_values) - 1
        plot!(p2, start_idx:end_idx, generated_values, 
              label="Generated (T=$(temp))", lw=2, alpha=0.8)
    end
    
    display(p2)
end

println("\n=== Autoregressive Transformer Example Completed ===")

