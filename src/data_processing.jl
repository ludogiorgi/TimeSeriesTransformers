"""
# Data Processing Utilities

Functions for creating and managing datasets for time series transformer training,
including autoregressive sequence preparation and batch tensor conversion.
"""

using Flux

# Import from parent module to access processor functions
import ..TimeSeriesProcessor

export get_sequence_dataset, create_autoregressive_batch_tensor

"""
    get_sequence_dataset(processor, sequence_length)

Create autoregressive training dataset where each sequence predicts the next token at each position.
Returns sequences prepared for autoregressive transformer training.
"""
function get_sequence_dataset(processor::TimeSeriesProcessor, sequence_length::Int)
    sequences = processor.sequences
    vocab_size = processor.vocab_size
    
    X = Vector{Vector{Int}}()
    y = Vector{Vector{Int}}()  # Changed to return sequence targets instead of single values
    
    for seq in sequences
        if length(seq) >= sequence_length + 1  # Need +1 for autoregressive targets
            for i in 1:(length(seq) - sequence_length)
                # Input sequence
                input_seq = seq[i:(i + sequence_length - 1)]
                # Target sequence (shifted by one position)
                target_seq = seq[(i + 1):(i + sequence_length)]
                
                push!(X, input_seq)
                push!(y, target_seq)
            end
        end
    end
    
    return X, y
end

"""
    create_autoregressive_batch_tensor(X_batch, y_batch, vocab_size)

Convert batch of sequences to tensors for autoregressive training.
Returns properly formatted tensors for the transformer model.
"""
function create_autoregressive_batch_tensor(X_batch::Vector{Vector{Int}}, y_batch::Vector{Vector{Int}}, vocab_size::Int)
    batch_size = length(X_batch)
    seq_len = length(X_batch[1])
    
    # Create input tensor: (vocab_size, sequence_length, batch_size)
    X_tensor = zeros(Float32, vocab_size, seq_len, batch_size)
    
    # Create target tensor: (sequence_length, batch_size)
    y_tensor = zeros(Int, seq_len, batch_size)
    
    for i in 1:batch_size
        # Convert input sequence to one-hot
        for t in 1:seq_len
            X_tensor[X_batch[i][t], t, i] = 1.0f0
        end
        
        # Set target sequence
        y_tensor[:, i] = y_batch[i]
    end
    
    return X_tensor, y_tensor
end