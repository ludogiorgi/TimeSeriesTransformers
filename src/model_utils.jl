"""
# Model Utilities

Utility functions for autoregressive prediction and sequence generation using the trained transformer model.
"""

using Flux
using Statistics
using Random

# Import from parent module to access transformer functions
import ..USE_THREADING, ..set_threading

export predict_next_token, generate_sequence, sample_from_distribution, predict_sequence_continuation

"""
    predict_next_token(model, input_sequence, vocab_size)

Predict the next token given an input sequence using autoregressive generation.
Returns the predicted token index and probability distribution.
"""
function predict_next_token(model::TransformerModel, input_sequence::Vector{Int}, vocab_size::Int)
    # Disable threading for inference
    old_threading = USE_THREADING[]
    set_threading(false)
    
    seq_len = length(input_sequence)
    
    # Convert to one-hot tensor: (vocab_size, sequence_length, 1)
    X = zeros(Float32, vocab_size, seq_len, 1)
    for t in 1:seq_len
        X[input_sequence[t], t, 1] = 1.0f0
    end
    
    # Forward pass
    output = model(X)  # Shape: (vocab_size, sequence_length, 1)
    
    # Get prediction from the last position
    logits = output[:, end, 1]  # Shape: (vocab_size,)
    probs = softmax(logits)
    
    # Get predicted token
    predicted_token = argmax(probs)
    
    # Restore threading
    set_threading(old_threading)
    
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