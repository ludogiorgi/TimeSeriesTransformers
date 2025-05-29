"""
# Prediction Utilities for Transformer Models

This module provides utility functions for making predictions and generating time series
using trained transformer models.

## Functions

- `predict`: Make a single-step prediction from a sequence
- `generate`: Generate future trajectories from multiple initial conditions
- `ensemble_predict`: Generate ensemble trajectories with fixed seeds
- `compute_validation_accuracy`: Compute validation accuracy
"""

using Flux: softmax
using Random

"""
    predict(model, sequence)

Make a prediction for a single sequence.

# Arguments
- `model::TransformerModel`: Trained transformer model
- `sequence`: Input sequence (Vector{Int} or batch)

# Returns
- `Int`: Predicted next cluster index
"""
function predict(model, sequence)
    # Check if the sequence is a single vector of integers
    if isa(sequence, Vector{<:Integer})
        # Wrap it in a Vector to create a batch of size 1
        input = [sequence]
    else
        # Otherwise, assume it's already properly batched
        input = sequence
    end
    
    # Get model output
    output = model(input)
    
    # Apply softmax to get probabilities
    probs = softmax(output, dims=1)
    
    # Find the predicted class
    _, predicted_idx = findmax(probs)
    return predicted_idx[1]
end

"""
    generate(model, initial_conditions, T)

Generate future trajectories from multiple initial conditions using autoregressive prediction.

# Arguments
- `model`: Trained transformer model
- `initial_conditions`: Vector of initial sequences (each a Vector{Int})
- `T::Int`: Number of time steps to generate into the future

# Returns
- `Vector{Vector{Int}}`: Generated trajectories, one for each initial condition

# Example
```julia
# Generate 10 steps from 3 different starting points
initial_seqs = [X_val[1], X_val[50], X_val[100]]
trajectories = generate(model, initial_seqs, 10)
```
"""
function generate(model, initial_conditions::Vector{<:Vector{<:Integer}}, T::Int)
    trajectories = Vector{Vector{Int}}()
    
    # Generate trajectory for each initial condition
    for initial_seq in initial_conditions
        trajectory = Int[]
        current_sequence = copy(initial_seq)
        
        # Generate T future steps
        for step in 1:T
            # Predict next cluster
            next_cluster = predict(model, current_sequence)
            push!(trajectory, next_cluster)
            
            # Update sequence: remove first element and add prediction
            current_sequence = vcat(current_sequence[2:end], next_cluster)
        end
        
        push!(trajectories, trajectory)
    end
    
    return trajectories
end

"""
    ensemble_predict(model, processor, initial_sequence, n_ens::Int, pred_steps::Int, ensemble_seeds, seed_offset::Int)

Generate ensemble trajectories using fixed random seeds for reproducibility.

# Arguments
- `model`: Trained transformer model
- `processor`: Data processor with cluster centers
- `initial_sequence`: Initial sequence to start the prediction
- `n_ens::Int`: Number of ensemble members to generate
- `pred_steps::Int`: Number of prediction steps
- `ensemble_seeds`: Fixed seeds for randomness
- `seed_offset::Int`: Offset to use with the fixed seeds

# Returns
- `trajectories`: Generated cluster index trajectories
- `values_trajectories`: Generated value trajectories corresponding to the cluster indices
"""
function ensemble_predict(model, processor, initial_sequence, n_ens::Int, pred_steps::Int, ensemble_seeds, seed_offset::Int)
    trajectories = zeros(Int, pred_steps, n_ens)
    values_trajectories = zeros(Float32, pred_steps, n_ens)
    
    # Generate each ensemble member
    for ens in 1:n_ens
        current_sequence = copy(initial_sequence)
        
        for step in 1:pred_steps
            # Use fixed seed for this specific ensemble member and step
            seed_idx = seed_offset + (ens-1) * pred_steps + step
            if seed_idx <= length(ensemble_seeds)
                Random.seed!(ensemble_seeds[seed_idx])
            end
            
            # Convert sequence to proper autoregressive format for model input
            vocab_size = processor.vocab_size
            seq_len = length(current_sequence)
            X_ar = zeros(Float32, vocab_size, seq_len, 1)
            for pos in 1:seq_len
                X_ar[current_sequence[pos], pos, 1] = 1.0f0
            end
            
            # Get probability distribution from transformer
            prob_output = model(X_ar)
            probs = softmax(prob_output[:, end, 1])  # Get probs for last position
            
            # Sample from the probability distribution with fixed seed
            rand_val = rand()
            cumsum_probs = cumsum(probs)
            next_cluster = findfirst(cumsum_probs .>= rand_val)
            
            if next_cluster === nothing
                next_cluster = length(probs)
            end
            
            # Store the prediction
            trajectories[step, ens] = next_cluster
            values_trajectories[step, ens] = processor.cluster_centers[next_cluster]
            
            # Update sequence for next prediction
            current_sequence = vcat(current_sequence[2:end], next_cluster)
        end
    end
    
    return trajectories, values_trajectories
end

"""
    compute_validation_accuracy(model, X_val, y_val; verbose=true)

Compute validation accuracy for the model.

# Arguments
- `model`: Trained transformer model
- `X_val`: Validation input sequences
- `y_val`: Validation target values
- `verbose::Bool`: Whether to print detailed results

# Returns
- `accuracy::Float64`: Classification accuracy
- `predictions::Vector{Int}`: Model predictions for all validation samples
"""
function compute_validation_accuracy(model, X_val, y_val; verbose::Bool=true)
    correct = 0
    total = length(X_val)
    predictions = Int[]
    
    # Get vocab size from processor if available, otherwise estimate
    vocab_size = 10  # Default fallback
    try
        # Try to get all unique values to estimate vocab size
        all_values = unique(vcat(X_val...))
        vocab_size = maximum(all_values)
    catch
        vocab_size = 10
    end
    
    for i in 1:total
        try
            # Convert sequence to proper autoregressive format for model input
            sequence = X_val[i]
            seq_len = length(sequence)
            X_ar = zeros(Float32, vocab_size, seq_len, 1)
            for pos in 1:seq_len
                if sequence[pos] >= 1 && sequence[pos] <= vocab_size
                    X_ar[sequence[pos], pos, 1] = 1.0f0
                end
            end
            
            # Use deterministic prediction
            prob_output = model(X_ar)
            probs = softmax(prob_output[:, end, 1])  # Get probs for last position
            pred = argmax(probs)
            push!(predictions, pred)
            if pred == y_val[i]
                correct += 1
            end
        catch e
            # If prediction fails, just add a random prediction
            push!(predictions, 1)
        end
    end
    
    accuracy = correct / total
    
    if verbose
        println("Current validation accuracy: $(round(accuracy * 100, digits=2))%")
    end
    
    return accuracy, predictions
end

"""
    predict_next_token(model, input_sequence, vocab_size)

Predict the next token given an input sequence using autoregressive generation.
Returns the predicted token index and probability distribution.
"""
function predict_next_token(model::TransformerModel, input_sequence::Vector{Int}, vocab_size::Int)
    seq_len = length(input_sequence)
    
    # Convert to one-hot tensor: (vocab_size, sequence_length, 1)
    X = zeros(Float32, vocab_size, seq_len, 1)
    for t in 1:seq_len
        if input_sequence[t] >= 1 && input_sequence[t] <= vocab_size
            X[input_sequence[t], t, 1] = 1.0f0
        end
    end
    
    # Forward pass
    output = model(X)  # Shape: (vocab_size, sequence_length, 1)
    
    # Get prediction from the last position
    logits = output[:, end, 1]  # Shape: (vocab_size,)
    probs = softmax(logits)
    
    # Get predicted token
    predicted_token = argmax(probs)
    
    return predicted_token, probs
end

"""
    generate_sequence(model, seed_sequence, vocab_size, n_generate; temperature=1.0)

Generate a sequence autoregressively using the trained transformer model.
"""
function generate_sequence(model::TransformerModel, seed_sequence::Vector{Int}, 
                         vocab_size::Int, n_generate::Int; temperature::Float32=1.0f0)
    generated = copy(seed_sequence)
    
    for _ in 1:n_generate
        # Predict next token
        next_token, probs = predict_next_token(model, generated, vocab_size)
        
        # Apply temperature sampling if specified
        if temperature != 1.0f0
            scaled_logits = log.(probs .+ 1e-8) ./ temperature
            probs = softmax(scaled_logits)
        end
        
        # Sample from the distribution (or take argmax for greedy)
        if temperature > 0.0f0
            next_token = sample_from_distribution(probs)
        end
        
        push!(generated, next_token)
    end
    
    return generated
end

"""
    sample_from_distribution(probs)

Sample a token index from a probability distribution.
"""
function sample_from_distribution(probs::AbstractVector)
    cumprobs = cumsum(probs)
    r = rand()
    for i in 1:length(cumprobs)
        if r <= cumprobs[i]
            return i
        end
    end
    return length(probs)  # Fallback
end

"""
    predict_sequence_continuation(model, processor, input_sequence, n_predict; temperature=1.0)

Predict continuation of a sequence using the trained transformer model.
"""
function predict_sequence_continuation(model::TransformerModel, processor::TimeSeriesProcessor, 
                                     input_sequence::Vector{Int}, n_predict::Int; 
                                     temperature::Float32=1.0f0)
    vocab_size = processor.vocab_size
    return generate_sequence(model, input_sequence, vocab_size, n_predict; temperature=temperature)
end

# Update existing functions to work with autoregressive format
"""
    predict(model, processor, X_test; n_samples=100)

Updated predict function that works with autoregressive model.
"""
function predict(model::TransformerModel, processor::TimeSeriesProcessor, X_test; n_samples::Int=100)
    vocab_size = processor.vocab_size
    predictions = Vector{Vector{Int}}()
    
    n_predict = min(n_samples, length(X_test))
    
    for i in 1:n_predict
        input_seq = X_test[i]
        # Generate next token prediction
        predicted_token, _ = predict_next_token(model, input_seq, vocab_size)
        push!(predictions, [predicted_token])
    end
    
    return predictions
end