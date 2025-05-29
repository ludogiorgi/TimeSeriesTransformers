using Revise
using Flux
using Statistics
using Plots
using Random
using Zygote
using TimeSeriesTransformers

function transformer_loss(model::TransformerModel, X::AbstractArray, y::AbstractArray; l2_reg::Float32=0.0f0)
    
    # X shape: (vocab_size, sequence_length, batch_size)
    # y shape: (sequence_length, batch_size) - targets for each position
    
    # Forward pass
    output = model(X)  # Shape: (vocab_size, sequence_length, batch_size)
    
    # Apply softmax to get probabilities
    probs = softmax(output, dims=1)  # Shape: (vocab_size, sequence_length, batch_size)
    
    # Prepare targets - each position predicts the next token
    # For position i, we predict y[i+1], so we need to shift targets
    seq_len = size(X, 2)
    batch_size = size(X, 3)
    vocab_size = size(output, 1)
    
    total_loss = Float32(0)
    valid_predictions = 0
    
    # Compute loss for each position (except the last one, which has no target)
    for pos in 1:(seq_len-1)
        # Get predictions at position pos
        pred_logits = output[:, pos, :]  # Shape: (vocab_size, batch_size)
        pred_probs = probs[:, pos, :]    # Shape: (vocab_size, batch_size)
        
        # Get targets (next tokens) - y[pos+1] for each batch
        targets = y[pos+1, :]  # Shape: (batch_size,)
        
        # Convert targets to one-hot encoding
        target_onehot = Flux.onehotbatch(targets, 1:vocab_size)  # Shape: (vocab_size, batch_size)
        
        # Compute cross-entropy loss for this position
        pos_loss = -mean(sum(target_onehot .* log.(pred_probs .+ Float32(1e-8)), dims=1))
        
        total_loss += pos_loss
        valid_predictions += 1
    end
    
    # Average loss across all valid predictions
    if valid_predictions > 0
        total_loss = total_loss / valid_predictions
    end
    
    return total_loss
end

data = generate_lorenz63_data(20000, tspan=(0.0, 2000.0))
y_data = data[:, 2]
num_clusters = 10

##

processor = TimeSeriesProcessor(y_data, num_clusters)

sequence_length = 64    # Reduced sequence length
d_model = 32           # Model dimension
num_heads = 8          # Number of attention heads
num_layers = 1         # Reduced number of transformer layers
dropout_rate = 0.1     # Dropout rate for regularization

# Validate configuration
head_dim = d_model ÷ num_heads

model = TransformerModel(
    num_clusters = num_clusters,
    latent_dim = num_clusters,
    d_model = d_model,
    num_heads = num_heads,
    num_layers = num_layers,
    dropout_rate = dropout_rate
)


# Get sequence data
X, y = get_sequence_dataset(processor, sequence_length)
processor.vocab_size 
processor.cluster_assignments
    
# Create train/validation split
train_split = 0.8
n_samples = length(X)
train_idx = 1:Int(floor(train_split * n_samples))
val_idx = (train_idx[end] + 1):n_samples
    
X_train = X[train_idx]
y_train = y[train_idx]
X_val = X[val_idx]
y_val = y[val_idx]

start_epoch = 1
epochs = 100
batch_size = 16      # Increased batch size to process fewer total batches
learning_rate = 1f-3
n_batches_per_epoch = 100  # Increased to handle larger dataset efficiently

all_batch_indices = [i:min(i+batch_size-1, length(X_train)) 
                        for i in 1:batch_size:length(X_train)]

max_batches = min(n_batches_per_epoch, length(all_batch_indices))
vocab_size = processor !== nothing ? processor.vocab_size : 10  # fallback
    
# Get trainable parameters for optimization
ps = Flux.params(model)

    
# Setup Adam optimizer with specified learning rate
opt = Adam(learning_rate)
    
# Main training loop
for epoch in start_epoch:epochs

    start_epoch = 1
    train_losses = Float32[]
    val_losses = Float32[]

    # Training phase - randomly sample batches for this epoch
    sampled_batch_indices = Random.shuffle(all_batch_indices)[1:max_batches]
        
    epoch_loss = 0.0
    batch_count = 0
        
    for idx in sampled_batch_indices
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        # Prepare autoregressive batch data
        X_ar, y_ar = prepare_autoregressive_batch(X_batch, y_batch, vocab_size)   
        # Calculate gradients using the autoregressive loss function
        loss, gs = Flux.withgradient(ps) do
            transformer_loss(model, X_ar, y_ar; l2_reg=0.0f0)
        end
        # Update model parameters
        Flux.Optimise.update!(opt, ps, gs)
            
        epoch_loss += loss
        batch_count += 1
    end
        
    # Calculate average training loss for this epoch
    avg_train_loss = epoch_loss / batch_count
    push!(train_losses, avg_train_loss)
        
    # Validation phase using the same loss function
    X_val_ar, y_val_ar = prepare_autoregressive_batch(X_val, y_val, vocab_size)
    val_loss = transformer_loss(model, X_val_ar, y_val_ar; l2_reg=0.0f0)
    push!(val_losses, val_loss)
    println("Epoch $epoch: Train Loss = $avg_train_loss, Val Loss = $val_loss")
end

plot(train_loss, label="Training Loss", color=:blue, title="Training and Validation Loss")
plot!(val_losses, label="Validation Loss", color=:red)

##
count = 0
for i in 1:1000
idx = rand(1:length(X_val))

X_batch, y_batch = X_train[idx:idx], y_train[idx:idx]
X_ar, y_ar = prepare_autoregressive_batch(X_batch, y_batch, vocab_size) 

pred = findmax(softmax(model(X_ar)[:,end,1]))[2]
truth = y_ar[end,1]
if pred == truth
    count += 1
end
end
println("Accuracy on random samples: $(count / 1000 * 100)%")

##

# Multi-step prediction function
function predict_future_steps(model, X_input::Vector{Int}, T::Int, vocab_size::Int)
    """
    Predict T timesteps into the future using autoregressive generation
    X_input: Initial sequence as vector of cluster indices
    T: Number of timesteps to predict
    vocab_size: Size of vocabulary (number of clusters)
    """
    
    # Start with the input sequence
    current_seq = copy(X_input)
    predictions = Int[]
    
    for t in 1:T
        # Prepare current sequence for model input
        X_ar, _ = prepare_autoregressive_batch([current_seq], [current_seq], vocab_size)
        
        # Get model prediction for the last position
        output = model(X_ar)  # Shape: (vocab_size, seq_len, 1)
        pred_logits = output[:, end, 1]  # Get last position prediction
        
        # Get most likely next token
        pred_probs = softmax(pred_logits)
        next_token = findmax(pred_probs)[2]  # Use findmax like in accuracy test
        
        push!(predictions, next_token)
        
        # Update sequence by shifting and adding new prediction
        current_seq = vcat(current_seq[2:end], [next_token])
    end
    
    return predictions
end

# Convert cluster indices back to original values
function clusters_to_values(cluster_indices::Vector{Int}, processor::TimeSeriesProcessor, original_data::Vector{Float64})
    """Convert cluster indices back to original time series values using cluster centroids"""
    values = Float64[]
    for cluster_idx in cluster_indices
        # Find the centroid value for this cluster
        cluster_mask = processor.cluster_assignments .== cluster_idx
        if any(cluster_mask)
            cluster_values = original_data[cluster_mask]
            centroid = mean(cluster_values)
            push!(values, centroid)
        else
            # Fallback if cluster is empty
            push!(values, mean(original_data))
        end
    end
    return values
end

# Perform multi-step prediction
T = 50  # Number of timesteps to predict into the future
test_idx = rand(1:length(X_val))  # Random validation sequence
test_sequence = X_val[test_idx]
test_targets = y_val[test_idx]

println("Performing $T-step prediction...")

# Use first part of sequence as input, predict the rest
input_length = min(32, length(test_sequence) - T)  # Use 32 timesteps as input
input_seq = test_sequence[1:input_length]
true_future = test_targets[(input_length+1):(input_length+T)]

# Generate predictions
predicted_clusters = predict_future_steps(model, input_seq, T, vocab_size)

# Convert to original values
predicted_values = clusters_to_values(predicted_clusters, processor, y_data)
true_values = clusters_to_values(true_future, processor, y_data)
input_values = clusters_to_values(input_seq, processor, y_data)

# Create visualization
time_input = 1:input_length
time_future = (input_length+1):(input_length+T)

p = plot(time_input, input_values, 
         label="Input Sequence", 
         linewidth=2, 
         color=:black,
         title="Multi-step Prediction ($T steps ahead)")

plot!(p, time_future, true_values, 
      label="True Future", 
      linewidth=2, 
      color=:blue)

plot!(p, time_future, predicted_values, 
      label="Predicted Future", 
      linewidth=2, 
      color=:red,
      linestyle=:dash)

# Add vertical line to separate input from prediction
vline!(p, [input_length + 0.5], 
       color=:gray, 
       linestyle=:dot, 
       linewidth=1, 
       label="Prediction Start")

xlabel!(p, "Time Steps")
ylabel!(p, "Value")
display(p)

# Calculate prediction metrics
mse = mean((predicted_values .- true_values).^2)
mae = mean(abs.(predicted_values .- true_values))
cluster_accuracy = mean(predicted_clusters .== true_future)

println("\n$T-Step Prediction Metrics:")
println("MSE (values): $mse")
println("MAE (values): $mae") 
println("Cluster Accuracy: $(cluster_accuracy * 100)%")

##
