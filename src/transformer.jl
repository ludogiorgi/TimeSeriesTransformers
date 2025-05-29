"""
# Transformer Architecture Implementation

This module provides a complete implementation of the Transformer architecture as described in
"Attention Is All You Need" (Vaswani et al., 2017), optimized for time series prediction tasks.

## Architecture Overview

The implementation includes all core components of the transformer model:

- **Multi-Head Attention**: Parallel attention mechanisms with configurable head dimensions
- **Positional Encoding**: Sinusoidal position embeddings for sequence modeling  
- **Feed-Forward Networks**: Position-wise fully connected layers with ReLU activation
- **Layer Normalization**: Stabilizes training and improves convergence
- **Residual Connections**: Enables training of deep networks

## Key Features

- **Flexible Head Dimensions**: Support for head dimensions independent of model dimension
- **Causal Masking**: Prevents attention to future positions for autoregressive prediction
- **Optimized Performance**: Efficient tensor operations with optional multi-threading
- **Gradient Compatibility**: Full support for automatic differentiation with Zygote.jl

## Mathematical Foundation

The transformer uses scaled dot-product attention:
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

Multi-head attention combines multiple attention heads:
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

## Usage Examples

```julia
# Create transformer model
model = TransformerModel(
    num_clusters=10, latent_dim=10, d_model=64,
    num_heads=8, num_layers=2, dropout_rate=0.1
)

# Forward pass
sequences = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]]
output = model(sequences)

# Make predictions
prediction = predict(model, [1, 2, 3, 4, 5])
```

## Performance Optimizations

- **Mask Caching**: Causal masks are cached for improved performance
- **Threading Control**: Conditional multi-threading with gradient compatibility
- **Memory Efficiency**: Optimized tensor operations and minimal allocations

## References

- Vaswani, A., et al. (2017). "Attention Is All You Need". NIPS.
- Ba, J.L., Kiros, J.R., & Hinton, G.E. (2016). "Layer Normalization". arXiv preprint.
"""

import Flux
import Flux: Dense, Dropout, LayerNorm, softmax
using LinearAlgebra
using Random
using Statistics

# Add threading capabilities
using Base.Threads

# Cache for storing precomputed causal masks
const CAUSAL_MASK_CACHE = Dict{Int, Matrix{Float32}}()

# Global flag to control threading (disabled during gradient computation)
const USE_THREADING = Ref(true)

"""
    set_threading(enabled::Bool)

Enable or disable threading in the transformer. Threading should be disabled
during gradient computation with Zygote.
"""
function set_threading(enabled::Bool)
    USE_THREADING[] = enabled
end

"""
    create_causal_mask(seq_len)

Create a causal attention mask that prevents attending to future positions.
Uses a cache for improved performance.
"""
function create_causal_mask(seq_len)
    # Check if mask is already cached
    if haskey(CAUSAL_MASK_CACHE, seq_len)
        return CAUSAL_MASK_CACHE[seq_len]
    end
    
    # Create row indices and column indices for broadcasting
    row_indices = reshape(1:seq_len, seq_len, 1)
    col_indices = reshape(1:seq_len, 1, seq_len)
    
    # Create the mask directly using broadcasting
    mask = ifelse.(row_indices .>= col_indices, 0.0f0, Float32(-Inf))
    
    # Cache the mask for future use
    CAUSAL_MASK_CACHE[seq_len] = mask
    return mask
end

"""
    MultiHeadAttention

Multi-head self-attention layer as described in the transformer architecture.
Modified to support num_heads > d_model with explicit head dimension.
"""
struct MultiHeadAttention
    num_heads::Int
    head_dim::Int
    W_q::Dense
    W_k::Dense
    W_v::Dense
    W_o::Dense
    dropout::Dropout
    scale::Float32
    
    function MultiHeadAttention(d_model::Int, num_heads::Int, head_dim::Int=0, dropout_rate::Float64=0.1)
        # If head_dim is not provided or is 0, calculate it from d_model (backward compatibility)
        actual_head_dim = head_dim <= 0 ? div(d_model, num_heads) : head_dim
        
        # No longer require d_model to be divisible by num_heads
        if head_dim <= 0 && d_model % num_heads != 0
            @warn "d_model ($d_model) is not divisible by num_heads ($num_heads). Using head_dim = $(actual_head_dim)"
        end
        
        scale = Float32(1 / sqrt(actual_head_dim))
        
        # Project from d_model to (num_heads * head_dim) for queries, keys, and values
        total_dim = num_heads * actual_head_dim
        W_q = Dense(d_model, total_dim)
        W_k = Dense(d_model, total_dim)
        W_v = Dense(d_model, total_dim)
        
        # Project back from (num_heads * head_dim) to d_model
        W_o = Dense(total_dim, d_model)
        
        dropout = Dropout(dropout_rate)
        
        new(num_heads, actual_head_dim, W_q, W_k, W_v, W_o, dropout, scale)
    end
end

# Add Flux functor declaration for MultiHeadAttention
Flux.@functor MultiHeadAttention

"""
    (mha::MultiHeadAttention)(query::AbstractArray; mask=nothing)

Self-attention case (query = key = value).
"""
function (mha::MultiHeadAttention)(query::AbstractArray; mask=nothing)
    return mha(query, query, query; mask=mask)
end

"""
    (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray; mask=nothing)

Key-value attention with same key and value.
"""
function (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray; mask=nothing)
    return mha(query, key, key; mask=mask)
end

"""
    (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray, value::AbstractArray; mask=nothing)

Multi-head attention implementation supporting arbitrary head dimensions independent of d_model.
"""
function (mha::MultiHeadAttention)(query::AbstractArray, key::AbstractArray, value::AbstractArray; mask=nothing)
    # Get dimensions
    d_model, seq_len_q, batch_size = size(query)
    _, seq_len_k, _ = size(key)
    
    # Get head dimensions
    h = mha.num_heads
    d_k = mha.head_dim
    total_dim = h * d_k
    
    # Project inputs to Q, K, V spaces - flatten once for efficiency
    q_flat = reshape(query, d_model, :)
    k_flat = reshape(key, d_model, :)
    v_flat = reshape(value, d_model, :)
    
    q_proj_flat = mha.W_q(q_flat)
    k_proj_flat = mha.W_k(k_flat)
    v_proj_flat = mha.W_v(v_flat)
    
    # Reshape projections to include head dimension
    q_proj = reshape(q_proj_flat, total_dim, seq_len_q, batch_size)
    k_proj = reshape(k_proj_flat, total_dim, seq_len_k, batch_size)
    v_proj = reshape(v_proj_flat, total_dim, seq_len_k, batch_size)
    
    # Process each batch using functional programming
    batch_outputs = map(1:batch_size) do b
        q_batch = view(q_proj, :, :, b)
        k_batch = view(k_proj, :, :, b)
        v_batch = view(v_proj, :, :, b)
        
        # Process heads - always use sequential processing for Zygote compatibility
        head_outputs = map(1:h) do head
            head_start = (head-1) * d_k + 1
            head_end = head * d_k
            
            q_head = view(q_batch, head_start:head_end, :)
            k_head = view(k_batch, head_start:head_end, :)
            v_head = view(v_batch, head_start:head_end, :)
            
            # Compute scaled dot-product attention
            scores = (q_head' * k_head) .* mha.scale
            
            # Apply mask if provided
            if mask !== nothing
                scores = scores .+ mask
            end
            
            # Apply softmax to get attention weights
            attention_weights = softmax(scores, dims=2)
            attention_weights = mha.dropout(attention_weights)
            
            # Apply attention to values
            v_head * attention_weights'
        end
        
        # Concatenate all heads for this batch (non-mutating)
        vcat(head_outputs...)
    end
    
    # Combine all batches (non-mutating)
    output_3d = cat(map(batch -> reshape(batch, total_dim, seq_len_q, 1), batch_outputs)..., dims=3)
    
    # Final projection from total_dim back to d_model
    output_flat = reshape(output_3d, total_dim, :)
    final_output_flat = mha.W_o(output_flat)
    
    return reshape(final_output_flat, d_model, seq_len_q, batch_size)
end

"""
    PositionalEncoding

Optimized transformer positional encoding layer with pre-computation.
"""
struct PositionalEncoding
    embedding::Matrix{Float32}
    
    function PositionalEncoding(max_len::Int, d_model::Int)
        # Pre-compute positional encoding matrix efficiently
        pe = zeros(Float32, d_model, max_len)
        position = reshape(1:max_len, 1, :)
        
        # Handle dimension calculation more carefully
        for i in 0:(d_model-1)
            # Calculate frequency based on position
            freq = exp(-(log(10000.0) * (i ÷ 2) / (d_model ÷ 2)))
            
            # Even indices get sine, odd indices get cosine
            if i % 2 == 0
                pe[i+1, :] = sin.(position .* freq)
            else
                pe[i+1, :] = cos.(position .* freq)
            end
        end
        
        new(pe)
    end
end

"""
    (pe::PositionalEncoding)(x)

Fully functional positional encoding application without mutations.
"""
function (pe::PositionalEncoding)(x)
    d_model, seq_len, batch_size = size(x)
    seq_len = min(seq_len, size(pe.embedding, 2))
    
    # Create result using functional approach (map + concatenate)
    result = cat(
        map(1:batch_size) do b
            # For each batch
            batch_result = cat(
                map(1:seq_len) do s
                    # For each position in sequence, add positional encoding
                    x[:, s, b] + pe.embedding[:, s]
                end...,
                dims=2
            )
            
            # If needed, pad with original values for positions beyond available encodings
            if seq_len < size(x, 2)
                cat(
                    batch_result,
                    x[:, (seq_len+1):end, b],
                    dims=2
                )
            else
                batch_result
            end
        end...,
        dims=3
    )
    
    return result
end

"""
    FeedForward

Standard feed-forward network used in transformer blocks.
"""
struct FeedForward
    W1::Dense
    W2::Dense
    dropout::Dropout
    
    function FeedForward(d_model::Int, d_ff::Int, dropout_rate::Float64=0.1)
        W1 = Dense(d_model, d_ff, relu)
        W2 = Dense(d_ff, d_model)
        dropout = Dropout(dropout_rate)
        
        new(W1, W2, dropout)
    end
end

# Add Flux functor declaration for FeedForward
Flux.@functor FeedForward

function (ff::FeedForward)(x::AbstractArray)
    # Get input dimensions
    d_model, seq_len, batch_size = size(x)
    
    # Reshape for dense layers
    x_flat = reshape(x, d_model, :)
    
    # Apply feed-forward network
    h = ff.W1(x_flat)
    h = ff.dropout(h)
    out = ff.W2(h)
    
    # Reshape back to original dimensions
    reshape(out, d_model, seq_len, batch_size)
end

"""
    TransformerEncoderLayer

Standard encoder layer that combines multi-head attention and feed-forward networks.
"""
struct TransformerEncoderLayer
    attention::MultiHeadAttention
    norm1::LayerNorm
    feed_forward::FeedForward
    norm2::LayerNorm
    dropout::Dropout
end

# Add Flux functor declaration for TransformerEncoderLayer
Flux.@functor TransformerEncoderLayer

function TransformerEncoderLayer(d_model::Int, num_heads::Int, d_ff::Int, dropout_rate::Float64=0.1, head_dim::Int=0)
    attention = MultiHeadAttention(d_model, num_heads, head_dim, dropout_rate)
    norm1 = LayerNorm(d_model)
    feed_forward = FeedForward(d_model, d_ff, dropout_rate)
    norm2 = LayerNorm(d_model)
    dropout = Dropout(dropout_rate)
    TransformerEncoderLayer(attention, norm1, feed_forward, norm2, dropout)
end

"""
    TransformerEncoder

Optimized stack of transformer encoder layers with positional encoding.
"""
struct TransformerEncoder
    pos_encoding::PositionalEncoding
    layers::Vector{TransformerEncoderLayer}
    norm::LayerNorm
end

# Add Flux functor declaration for TransformerEncoder
Flux.@functor TransformerEncoder

function TransformerEncoder(num_layers::Int, d_model::Int, num_heads::Int, d_ff::Int, 
                          max_seq_len::Int, dropout_rate::Float64=0.1, head_dim::Int=0)
    pos_encoding = PositionalEncoding(max_seq_len, d_model)
    layers = [TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate, head_dim) for _ in 1:num_layers]
    norm = LayerNorm(d_model)
    TransformerEncoder(pos_encoding, layers, norm)
end

function (encoder::TransformerEncoder)(x::AbstractArray, mask=nothing)
    # Add positional encoding
    x = encoder.pos_encoding(x)
    
    # Pass through encoder layers - this part is inherently sequential
    # but each layer can internally use parallelism
    for layer in encoder.layers
        x = layer(x, mask)
    end
    
    # Final layer normalization
    encoder.norm(x)
end

"""
    TransformerModel

Complete transformer model for sequence prediction.
"""
struct TransformerModel
    embedding::Dense  # Input embedding
    encoder::TransformerEncoder
    output::Dense  # Output projection
end

# Add Flux functor declaration for TransformerModel
Flux.@functor TransformerModel

function TransformerModel(;
    num_clusters::Int,  # renamed from input_dim
    latent_dim::Int,    # renamed from output_dim
    d_model::Int,
    num_heads::Int,
    num_layers::Int,
    max_seq_len::Int=1000,
    d_ff::Union{Int,Nothing}=nothing,
    dropout_rate::Float64=0.1,
    head_dim::Int=0    # New parameter for explicit head dimension
)
    # Set feed-forward dimension if not provided
    d_ff_actual = isnothing(d_ff) ? 4 * d_model : d_ff
    
    # Create layers
    embedding = Dense(num_clusters, d_model)
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff_actual, max_seq_len, dropout_rate, head_dim)
    output = Dense(d_model, latent_dim)
    
    TransformerModel(embedding, encoder, output)
end

# Forward pass
function (model::TransformerModel)(x; use_causal_mask=true)
    # Handle sequence input formats
    if isa(x, Vector{<:Vector{<:Integer}})
        x_embedded = process_sequence_input(x, model)
    else
        x_embedded = model.embedding(x)
    end
    
    # Get causal mask if needed - now using efficient caching
    mask = nothing
    if use_causal_mask
        seq_len = size(x_embedded, 2)
        mask = create_causal_mask(seq_len)
    end
    
    # Apply transformer encoder with mask
    encoded = model.encoder(x_embedded, mask)
    
    # For autoregressive training, we need predictions at all positions
    # Apply output projection to each position
    d_model, seq_len, batch_size = size(encoded)
    
    # Reshape to apply output layer: (d_model, seq_len * batch_size)
    encoded_flat = reshape(encoded, d_model, seq_len * batch_size)
    output_flat = model.output(encoded_flat)  # Shape: (output_dim, seq_len * batch_size)
    
    # Reshape back to sequence format: (output_dim, seq_len, batch_size)
    output_dim = size(output_flat, 1)
    reshape(output_flat, output_dim, seq_len, batch_size)
end

# Forward pass with residual connections and normalization
function (layer::TransformerEncoderLayer)(x::AbstractArray, mask=nothing)
    # Multi-head attention with residual connection and layer norm
    attended = layer.attention(x; mask=mask)
    attended = layer.dropout(attended)
    x1 = layer.norm1(x .+ attended)
    
    # Feed-forward with residual connection and layer norm
    transformed = layer.feed_forward(x1)
    transformed = layer.dropout(transformed)
    layer.norm2(x1 .+ transformed)
end

"""
    process_sequence_input(sequences, model)

Fully functional implementation for batch processing sequence inputs.
"""
function process_sequence_input(sequences::Vector{<:Vector{<:Integer}}, model::TransformerModel)
    batch_size = length(sequences)
    seq_len = maximum(length(seq) for seq in sequences)
    d_model = size(model.embedding.weight, 1)
    num_clusters = size(model.embedding.weight, 2)
    
    # Create result using purely functional approach
    batch_tensors = map(1:batch_size) do b
        seq = sequences[b]
        
        # Create sequence tensor using map
        cat(
            map(1:seq_len) do t
                if t <= length(seq)
                    # Ensure index is within bounds
                    cluster_idx = min(max(seq[t], 1), num_clusters)
                    model.embedding.weight[:, cluster_idx]
                else
                    zeros(Float32, d_model)
                end
            end...,
            dims=2
        )
    end
    
    # Combine into 3D tensor
    cat(map(tensor -> reshape(tensor, d_model, seq_len, 1), batch_tensors)..., dims=3)
end
