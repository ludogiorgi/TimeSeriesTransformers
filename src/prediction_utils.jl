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
            
            # Get probability distribution from transformer
            prob_output = model([current_sequence])
            probs = softmax(prob_output, dims=1)[:, 1]
            
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
    
    for i in 1:total
        # Use deterministic prediction
        prob_output = model([X_val[i]])
        probs = softmax(prob_output, dims=1)[:, 1]
        pred = argmax(probs)
        push!(predictions, pred)
        if pred == y_val[i]
            correct += 1
        end
    end
    
    accuracy = correct / total
    
    if verbose
        println("Current validation accuracy: $(round(accuracy * 100, digits=2))%")
    end
    
    return accuracy, predictions
end