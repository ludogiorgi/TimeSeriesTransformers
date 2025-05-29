"""
# Training Utilities

Comprehensive training infrastructure for transformer models including validation monitoring,
early stopping, model evaluation, and time series generation capabilities.
"""

using Flux
using Statistics
using Plots
using Random
using Zygote

# Import from parent module to access transformer functions
import ..USE_THREADING, ..set_threading

export train_with_validation!, train_and_evaluate!

"""
    train_with_validation!(model, X_train, y_train, X_val, y_val; kwargs...)

Train a transformer model with comprehensive validation monitoring, early stopping, and checkpointing.

# Arguments
- `model`: TransformerModel to train
- `X_train`: Training input sequences
- `y_train`: Training target values
- `X_val`: Validation input sequences  
- `y_val`: Validation target values

# Keyword Arguments
- `epochs::Int`: Number of training epochs (default: 200)
- `batch_size::Int`: Batch size for training (default: 32)
- `learning_rate::Float32`: Learning rate for optimizer (default: 1f-4)
- `early_stopping_patience::Int`: Early stopping patience (default: 5)
- `grad_clip::Float32`: Gradient clipping threshold (default: 1.0f0)
- `l2_reg::Float32`: L2 regularization coefficient (default: 0.0f0)
- `verbose::Bool`: Whether to print training progress (default: true)
- `plot_training::Bool`: Whether to plot training curves (default: true)
- `display_plots::Bool`: Whether to display plots during training (default: false)
- `checkpoint_dir::String`: Directory for saving checkpoints (default: "checkpoints")
- `checkpoint_frequency::Int`: Frequency of checkpointing (in epochs, default: 10)
- `resume_from_checkpoint::Bool`: Whether to resume from the latest checkpoint (default: false)
- `checkpoint_filename::String`: Base filename for checkpoints (default: "model_checkpoint")
- `analysis_callback_frequency::Int`: Frequency of analysis callback (in epochs, default: 0)
- `processor`: TimeSeriesProcessor for analysis callback (default: nothing)
- `y_data`: Additional data for analysis callback (default: nothing)

# Returns
- `(train_losses, val_losses)`: Training and validation loss history
"""
function train_with_validation!(model::TransformerModel, X_train, y_train, X_val, y_val;
                                epochs::Int=200,
                                batch_size::Int=32,
                                learning_rate::Float32=1f-4,
                                early_stopping_patience::Int=5,
                                grad_clip::Float32=1.0f0,
                                l2_reg::Float32=0.0f0,
                                verbose::Bool=true,
                                plot_training::Bool=true,
                                display_plots::Bool=false,
                                n_batches_per_epoch::Int=50,
                                checkpoint_dir::String="checkpoints",
                                checkpoint_frequency::Int=10,
                                resume_from_checkpoint::Bool=false,
                                checkpoint_filename::String="model_checkpoint",
                                analysis_callback_frequency::Int=0,
                                processor=nothing,
                                y_data=nothing)

    # Create checkpoint directory if it doesn't exist
    if !isdir(checkpoint_dir)
        mkpath(checkpoint_dir)
    end
    
    # Create figures directory in the project root if it doesn't exist
    project_root = dirname(dirname(@__FILE__))  # Go up from src/ to project root
    figures_dir = joinpath(project_root, "figures")
    if !isdir(figures_dir)
        mkpath(figures_dir)
    end
    
    checkpoint_path = joinpath(checkpoint_dir, "$(checkpoint_filename).jld2")
    training_state_path = joinpath(checkpoint_dir, "$(checkpoint_filename)_training_state.jld2")
    
    # Initialize training state
    start_epoch = 1
    train_losses = Float32[]
    val_losses = Float32[]
    best_val_loss = Inf
    patience_counter = 0
    best_model_state = nothing
    
    # Try to resume from checkpoint if requested
    if resume_from_checkpoint && isfile(training_state_path)
        try
            training_state = JLD2.load(training_state_path)
            start_epoch = training_state["epoch"] + 1
            train_losses = training_state["train_losses"]
            val_losses = training_state["val_losses"]
            best_val_loss = training_state["best_val_loss"]
            patience_counter = training_state["patience_counter"]
            
            if verbose
                println("Resuming training from epoch $start_epoch")
                println("Previous best validation loss: $(round(best_val_loss, digits=6))")
            end
            
            # Load the best model if it exists
            if isfile(checkpoint_path)
                # Note: We assume the processor is the same, so we only load model weights
                saved_data = JLD2.load(checkpoint_path)
                # Use string key instead of symbol key
                Flux.loadmodel!(model, saved_data["model_state"])  # Changed from [:model_state]
                best_model_state = deepcopy(model)
                if verbose
                    println("Loaded model weights from checkpoint")
                end
            end
        catch e
            @warn "Could not resume from checkpoint: $e"
            @warn "Starting training from scratch"
            start_epoch = 1
        end
    end
    
    # Create batch indices for efficient mini-batch processing
    all_batch_indices = [i:min(i+batch_size-1, length(X_train)) 
                        for i in 1:batch_size:length(X_train)]
    
    # Limit the number of batches per epoch to prevent it from being too large
    max_batches = min(n_batches_per_epoch, length(all_batch_indices))
    
    # Get trainable parameters for optimization
    ps = Flux.params(model)
    if verbose && start_epoch == 1
        param_count = sum(length, ps)
        println("Training model with $(param_count) parameters")
        println("Total available batches: $(length(all_batch_indices))")
        println("Batches per epoch: $max_batches")
        println("Batch size: $batch_size")
        println("Checkpoint frequency: every $checkpoint_frequency epochs")
        println("Checkpoint directory: $checkpoint_dir")
    end
    
    # Setup Adam optimizer with specified learning rate
    opt = Adam(learning_rate)
    
    # Initialize best model state if not loaded from checkpoint
    if best_model_state === nothing
        best_model_state = deepcopy(model)
    end
    
    if verbose
        println("Starting training with early stopping (patience: $early_stopping_patience)...")
    end
    
    # Initialize fixed analysis parameters once at the beginning if callback is enabled
    if analysis_callback_frequency > 0 && processor !== nothing && y_data !== nothing
        initialize_analysis_params!(X_val, y_val; reset=true)
        if verbose
            println("Fixed analysis parameters initialized for consistent callback results")
        end
    end
    
    # Main training loop
    for epoch in start_epoch:epochs
        # Training phase - randomly sample batches for this epoch
        sampled_batch_indices = Random.shuffle(all_batch_indices)[1:max_batches]
        
        epoch_loss = 0.0
        batch_count = 0
        
        for idx in sampled_batch_indices
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            
            # Prepare autoregressive batch data
            vocab_size = processor !== nothing ? processor.vocab_size : 10  # fallback
            X_ar, y_ar = prepare_autoregressive_batch(X_batch, y_batch, vocab_size)
            
            # Calculate gradients using the autoregressive loss function
            loss, gs = Flux.withgradient(ps) do
                transformer_loss(model, X_ar, y_ar; l2_reg=l2_reg)
            end
            
            # Apply gradient clipping if specified
            if grad_clip > 0
                for g in values(gs)
                    if g !== nothing
                        clamp!(g, -grad_clip, grad_clip)
                    end
                end
            end
            
            # Update model parameters
            Flux.Optimise.update!(opt, ps, gs)
            
            epoch_loss += loss
            batch_count += 1
        end
        
        # Calculate average training loss for this epoch
        avg_train_loss = epoch_loss / batch_count
        push!(train_losses, avg_train_loss)
        
        # Validation phase using the same autoregressive loss function
        vocab_size = processor !== nothing ? processor.vocab_size : 10  # fallback
        X_val_ar, y_val_ar = prepare_autoregressive_batch(X_val, y_val, vocab_size)
        val_loss = transformer_loss(model, X_val_ar, y_val_ar; l2_reg=l2_reg)
        push!(val_losses, val_loss)
        
        if verbose && (epoch ≤ 5 || epoch % 10 == 0)
            println("Epoch $epoch/$epochs: Train Loss = $(round(avg_train_loss, digits=6)), Val Loss = $(round(val_loss, digits=6))")
        end
        
        # Early stopping logic
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = deepcopy(model)
            if verbose && epoch > 1
                println("  → New best validation loss: $(round(val_loss, digits=6))")
            end
        else
            patience_counter += 1
            if verbose && epoch > 1
                println("  → No improvement ($patience_counter/$early_stopping_patience)")
            end
            
            # Stop training if patience exceeded
            if patience_counter >= early_stopping_patience
                if verbose
                    println("Early stopping triggered! Restoring best model...")
                end
                # Restore best model weights
                for (p1, p2) in zip(Flux.params(model), Flux.params(best_model_state))
                    p1 .= p2
                end
                break
            end
        end
        
        # Save checkpoint periodically
        if epoch % checkpoint_frequency == 0
            try
                # Create a dummy processor for saving (since we don't have access to the original)
                dummy_data = Float64.(1:10)  # Placeholder
                dummy_processor = TimeSeriesProcessor(dummy_data, 10)
                
                # Save model checkpoint - use string keys instead of symbols
                model_state = Flux.state(best_model_state)
                checkpoint_data = Dict(
                    "model_state" => model_state,  # Changed from :model_state
                    "version" => "1.0"             # Changed from :version
                )
                JLD2.save(checkpoint_path, checkpoint_data)
                
                # Save training state - already using string keys
                training_state = Dict(
                    "epoch" => epoch,
                    "train_losses" => train_losses,
                    "val_losses" => val_losses,
                    "best_val_loss" => best_val_loss,
                    "patience_counter" => patience_counter,
                    "learning_rate" => learning_rate,
                    "batch_size" => batch_size
                )
                JLD2.save(training_state_path, training_state)
                
                if verbose
                    println("  → Checkpoint saved at epoch $epoch")
                end
                
                # Call analysis callback if enabled and processor/data are provided
                if analysis_callback_frequency > 0 && epoch % analysis_callback_frequency == 0 && 
                   processor !== nothing && y_data !== nothing
                    try
                        # Save directly to project root figures directory
                        save_path = joinpath(figures_dir, "analysis_epoch_$(epoch).png")
                        fig = comprehensive_analysis_callback(model, processor, X_val, y_val, y_data; 
                                                            save_path=save_path, verbose=verbose)
                        if display_plots
                            display(fig)
                        end
                        if verbose
                            println("  → Analysis callback completed, saved to: $save_path")
                        end
                    catch callback_error
                        @warn "Analysis callback failed at epoch $epoch: $callback_error"
                        # Print more detailed error information
                        println("Detailed error: ")
                        showerror(stdout, callback_error)
                        println()
                    end
                end
                
            catch e
                @warn "Failed to save checkpoint: $e"
            end
        end
    end
    
    # Final checkpoint save
    try
        model_state = Flux.state(model)
        final_checkpoint_data = Dict(
            "model_state" => model_state,  # Changed from :model_state
            "version" => "1.0"             # Changed from :version
        )
        final_checkpoint_path = joinpath(checkpoint_dir, "$(checkpoint_filename)_final.jld2")
        JLD2.save(final_checkpoint_path, final_checkpoint_data)
        
        if verbose
            println("Final model saved to: $final_checkpoint_path")
        end
    catch e
        @warn "Failed to save final checkpoint: $e"
    end
    
    # Plot training history and save to figures folder
    if plot_training && length(train_losses) > 1
        p = plot(train_losses, label="Training", title="Transformer Learning Curve", 
                xlabel="Epoch", ylabel="Loss", legend=:topright)
        plot!(p, val_losses, label="Validation")
        
        # Save training curve to project root figures folder
        training_curve_path = joinpath(figures_dir, "training_curve.png")
        savefig(p, training_curve_path)
        if verbose
            println("Training curve saved to: $training_curve_path")
        end
        
        # Display only if requested
        if display_plots
            display(p)
        end
    end
    
    return train_losses, val_losses
end

"""
    train_and_evaluate!(model, processor, sequence_length; kwargs...)

Complete training and evaluation pipeline with checkpointing support.

# Arguments
- `model`: TransformerModel to train
- `processor`: TimeSeriesProcessor with data
- `sequence_length::Int`: Length of input sequences

# Keyword Arguments
- `train_split::Float64`: Fraction of data for training (default: 0.8)
- `checkpoint_dir::String`: Directory for saving checkpoints (default: "checkpoints")
- `checkpoint_frequency::Int`: Frequency of checkpointing (in epochs, default: 10)
- `resume_from_checkpoint::Bool`: Whether to resume from the latest checkpoint (default: false)
- `save_final_model::Bool`: Whether to save the final trained model (default: true)
- Other arguments passed to train_with_validation!

# Returns
- `(model, train_losses, val_losses)`
"""
function train_and_evaluate!(model::TransformerModel, processor::TimeSeriesProcessor, 
                             sequence_length::Int;
                             train_split::Float64=0.8,
                             verbose::Bool=true,
                             display_plots::Bool=false,  # New parameter
                             # Checkpoint parameters
                             checkpoint_dir::String="checkpoints",
                             checkpoint_frequency::Int=10,
                             resume_from_checkpoint::Bool=false,
                             save_final_model::Bool=true,
                             model_filename::String="trained_transformer",
                             kwargs...)
    
    if verbose
        println("Creating training sequences...")
    end
    
    # Get sequence data
    X, y = get_sequence_dataset(processor, sequence_length)
    
    # Create train/validation split
    n_samples = length(X)
    train_idx = 1:Int(floor(train_split * n_samples))
    val_idx = (train_idx[end] + 1):n_samples
    
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    
    if verbose
        println("Training samples: $(length(X_train))")
        println("Validation samples: $(length(X_val))")
    end
    
    # Train the model with checkpointing
    train_losses, val_losses = train_with_validation!(
        model, X_train, y_train, X_val, y_val; 
        checkpoint_dir=checkpoint_dir,
        checkpoint_frequency=checkpoint_frequency,
        resume_from_checkpoint=resume_from_checkpoint,
        display_plots=display_plots,  # Pass through the display parameter
        verbose=verbose, kwargs...)
    
    # Save final trained model with processor
    if save_final_model
        try
            final_model_path = joinpath(checkpoint_dir, "$(model_filename).jld2")
            # Use the save_model function that's already defined in the module
            save_model(model, processor, final_model_path)
            if verbose
                println("Complete model and processor saved to: $final_model_path")
            end
        catch e
            @warn "Failed to save final model: $e"
        end
    end
    
    return model, train_losses, val_losses
end

"""
    transformer_loss(model, X, y; l2_reg=0.0)

Compute autoregressive loss for transformer model with causal masking.
Each position in the sequence predicts the next token, with loss computed on all valid predictions.
Temporarily disables threading during forward pass for gradient compatibility.
"""
function transformer_loss(model::TransformerModel, X::AbstractArray, y::AbstractArray; l2_reg::Float32=0.0f0)
    # Disable threading during forward pass for gradient computation
    old_threading = USE_THREADING[]
    set_threading(false)
    
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
    
    # Add L2 regularization if specified
    if l2_reg > 0
        l2_penalty = Float32(0)
        for layer in model.encoder.layers
            # Regularize attention weights
            l2_penalty += sum(layer.attention.W_q.weight .^ 2)
            l2_penalty += sum(layer.attention.W_k.weight .^ 2)
            l2_penalty += sum(layer.attention.W_v.weight .^ 2)
            l2_penalty += sum(layer.attention.W_o.weight .^ 2)
            # Regularize feed-forward weights
            l2_penalty += sum(layer.feed_forward.W1.weight .^ 2)
            l2_penalty += sum(layer.feed_forward.W2.weight .^ 2)
        end
        total_loss += l2_reg * l2_penalty
    end
    
    # Restore threading state
    set_threading(old_threading)
    
    return total_loss
end

"""
    prepare_autoregressive_targets(y_sequence)

Prepare targets for autoregressive training where each position predicts the next token.
Input: sequence of shape (sequence_length,)
Output: input and target sequences for autoregressive training
"""
function prepare_autoregressive_targets(y_sequence::AbstractVector)
    seq_len = length(y_sequence)
    
    # Input sequence: all tokens except the last
    input_seq = y_sequence[1:end-1]
    
    # Target sequence: all tokens except the first  
    target_seq = y_sequence[2:end]
    
    return input_seq, target_seq
end

"""
    prepare_autoregressive_batch(X_batch, y_batch)

Prepare a batch for autoregressive training.
X_batch: Vector of input sequences (Vector{Vector{Int}})
y_batch: Vector of target sequences (Vector{Vector{Int}})
"""
function prepare_autoregressive_batch(X_batch::AbstractVector, y_batch::AbstractVector, vocab_size::Int)
    # If y_batch contains sequences (new format), use them directly
    if isa(y_batch[1], AbstractVector)
        batch_size = length(X_batch)
        seq_len = length(X_batch[1])
        
        return create_autoregressive_batch_tensor(X_batch, y_batch, vocab_size)
    else
        # Fallback for old format (single target values)
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
            
            # For targets, create autoregressive sequence
            # Shift input sequence by one position for targets
            y_tensor[1:end-1, i] = X_batch[i][2:end]
            y_tensor[end, i] = y_batch[i]  # Last target is the provided target
        end
        
        return X_tensor, y_tensor
    end
end
