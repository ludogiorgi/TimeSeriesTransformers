"""
# Analysis Callback for Comprehensive Model Evaluation

This module provides comprehensive analysis functionality for transformer models,
including ensemble predictions, phase space analysis, and statistical comparisons.
The analysis generates reproducible results by using fixed random seeds and starting points.
"""

using Plots
using Statistics
using StatsBase
using KernelDensity
using Flux: softmax
using Random

# Import prediction utilities
import ..ensemble_predict, ..compute_validation_accuracy

export comprehensive_analysis_callback, autocorrelation, initialize_analysis_params!

# Global dictionary to store fixed analysis parameters for reproducible results
const ANALYSIS_PARAMS = Dict{String, Any}()

"""
    initialize_analysis_params!(X_val, y_val; reset=false)

Initialize fixed parameters for consistent analysis across training epochs.
This ensures reproducible analysis by fixing random seeds and starting indices.

# Arguments
- `X_val`: Validation input sequences
- `y_val`: Validation target values
- `reset::Bool`: Whether to reset previously initialized parameters

# Note
This function should be called once before running multiple analyses to ensure
consistent comparison across different training epochs.
"""
function initialize_analysis_params!(X_val, y_val; reset=false)
    if reset || !haskey(ANALYSIS_PARAMS, "initialized")
        # Set fixed random seed for reproducible analysis
        Random.seed!(12345)
        
        # Store fixed starting indices for ensemble predictions (4 different validation points)
        max_start_idx = max(1, length(X_val) - 250)
        ANALYSIS_PARAMS["ensemble_start_indices"] = [
            rand(1:max_start_idx),
            rand(1:max_start_idx),
            rand(1:max_start_idx),
            rand(1:max_start_idx)
        ]
        
        # Fixed starting index for delayed prediction analysis
        delayed_start_idx = rand(1:max(1, min(length(X_val)-250, length(y_val)-250)))
        ANALYSIS_PARAMS["delayed_start_idx"] = delayed_start_idx
        
        # Fixed starting index for long trajectory generation
        ANALYSIS_PARAMS["long_traj_start_idx"] = rand(1:length(X_val))
        
        # Pre-generate all random seeds needed for ensemble predictions
        # This ensures deterministic ensemble generation across multiple runs
        n_ens = 50
        pred_steps = 20
        ensemble_seeds = rand(UInt32, n_ens * pred_steps * 4)  # 4 ensemble predictions
        ANALYSIS_PARAMS["ensemble_seeds"] = ensemble_seeds
        
        # Fixed seeds for long trajectory generation (1000 steps)
        long_traj_seeds = rand(UInt32, 1000)
        ANALYSIS_PARAMS["long_traj_seeds"] = long_traj_seeds
        
        ANALYSIS_PARAMS["initialized"] = true
        
        println("Analysis parameters initialized with fixed random seeds for reproducibility")
    end
end

"""
    autocorrelation(x, max_lag=50)

Calculate the autocorrelation function for a time series up to a maximum lag.

# Arguments
- `x`: Input time series vector
- `max_lag::Int`: Maximum lag to compute (default: 50)

# Returns
- Vector of autocorrelation values from lag 0 to max_lag

# Note
The autocorrelation at lag 0 is always 1.0 by definition.
"""
function autocorrelation(x, max_lag=50)
    n = length(x)
    x_centered = x .- mean(x)  # Center the time series
    acf = zeros(max_lag + 1)
    
    for lag in 0:max_lag
        if lag == 0
            acf[lag + 1] = 1.0  # Autocorrelation at lag 0 is always 1
        else
            # Calculate correlation between original and lagged series
            acf[lag + 1] = cor(x_centered[1:end-lag], x_centered[lag+1:end])
        end
    end
    
    return acf
end

"""
    comprehensive_analysis_callback(model, processor, X_val, y_val, y_data; 
                                   save_path=nothing, verbose=true)

Create comprehensive 6-row analysis for transformer model evaluation with fixed starting points.

This function generates a detailed analysis plot containing:
- Row 1-2: Four ensemble predictions from fixed validation points (2x2 grid)
- Row 3-5: Delayed prediction comparisons at t+5, t+10, t+20 steps
- Row 6: Statistical analysis comparing predicted vs. true distributions and autocorrelations

# Arguments
- `model`: Trained transformer model
- `processor`: TimeSeriesProcessor containing cluster information
- `X_val`: Validation input sequences
- `y_val`: Validation target cluster indices
- `y_data`: Original continuous time series data for statistical comparison

# Keyword Arguments
- `save_path::String`: Path to save the analysis plot (default: project_root/figures/comprehensive_analysis.png)
- `verbose::Bool`: Whether to print progress messages (default: true)

# Returns
- Plot object with comprehensive analysis

# Note
This function uses fixed random seeds and starting points for reproducible analysis.
Call `initialize_analysis_params!()` before first use to set up reproducible parameters.
"""
function comprehensive_analysis_callback(model, processor, X_val, y_val, y_data; 
                                       save_path=nothing,
                                       verbose=true)
    
    if verbose
        println("Creating comprehensive 6-row panel analysis...")
    end
    
    # Set default save path to project figures folder if not provided
    if save_path === nothing
        project_root = dirname(dirname(@__FILE__))  # Navigate from src/ to project root
        figures_dir = joinpath(project_root, "figures")
        save_path = joinpath(figures_dir, "comprehensive_analysis.png")
    end
    
    # Compute and display validation accuracy
    accuracy, predictions = compute_validation_accuracy(model, X_val, y_val; verbose=verbose)
    
    # Initialize fixed parameters for reproducible analysis if not already done
    initialize_analysis_params!(X_val, y_val)
    
    # Ensure output directory exists
    figures_dir = dirname(save_path)
    if !isdir(figures_dir)
        mkpath(figures_dir)
    end
    
    plots_array = []
    
    try
        # Rows 1-2: Four ensemble predictions from FIXED validation starting points
        if verbose
            println("Creating ensemble predictions from 4 fixed validation points...")
        end
        
        ensemble_start_indices = ANALYSIS_PARAMS["ensemble_start_indices"]
        ensemble_seeds = ANALYSIS_PARAMS["ensemble_seeds"]
        
        for i in 1:4
            # Use pre-determined fixed starting point for reproducibility
            start_idx = ensemble_start_indices[i]
            initial_sequence = copy(X_val[start_idx])
            
            # Generate ensemble prediction with pre-determined random seeds
            n_ens_small = 50
            pred_steps_small = 20
            seed_offset = (i-1) * n_ens_small * pred_steps_small
            
            trajectories_small, values_trajectories_small = ensemble_predict(
                model, processor, initial_sequence, n_ens_small, pred_steps_small, ensemble_seeds, seed_offset)
            
            # Calculate ensemble statistics (mean and confidence intervals)
            ensemble_mean_small = mean(values_trajectories_small, dims=2)[:, 1]
            ensemble_quantiles_small = [quantile(values_trajectories_small[j, :], [0.05, 0.25, 0.75, 0.95]) 
                                      for j in 1:pred_steps_small]
            
            # Get corresponding observed values for comparison
            obs_end_idx = min(start_idx + pred_steps_small - 1, length(y_val))
            observed_clusters_small = y_val[start_idx:obs_end_idx]
            observed_values_small = [processor.cluster_centers[cluster] for cluster in observed_clusters_small]
            
            # Pad with last value if sequence is shorter than prediction horizon
            while length(observed_values_small) < pred_steps_small
                push!(observed_values_small, observed_values_small[end])
            end
            
            # Extract quantiles for confidence interval plotting
            q05_small = [q[1] for q in ensemble_quantiles_small]
            q25_small = [q[2] for q in ensemble_quantiles_small]
            q75_small = [q[3] for q in ensemble_quantiles_small]
            q95_small = [q[4] for q in ensemble_quantiles_small]
            
            # Create ensemble prediction plot with confidence intervals
            p = plot(1:pred_steps_small, q05_small, fillto=q95_small, 
                     alpha=0.2, color=:blue, label="90% CI")
            plot!(p, 1:pred_steps_small, q25_small, fillto=q75_small, 
                  alpha=0.3, color=:blue, label="50% CI")
            plot!(p, 1:pred_steps_small, ensemble_mean_small, 
                  color=:red, linewidth=2, label="Ensemble Mean")
            plot!(p, 1:pred_steps_small, observed_values_small, 
                  color=:black, linewidth=2, label="Observed")
            plot!(p, title="Ensemble Prediction $i (Acc: $(round(accuracy*100, digits=1))%)", 
                  xlabel="Time Steps", ylabel="Value")
            
            push!(plots_array, p)
        end
        
        # Rows 3-5: Delayed prediction analysis from FIXED starting point
        if verbose
            println("Creating delayed prediction comparisons...")
        end
        
        # Use pre-determined fixed starting point for delayed predictions
        start_idx = ANALYSIS_PARAMS["delayed_start_idx"]
        
        # Extract true trajectory for comparison baseline
        max_seq_length = min(200, length(y_val) - start_idx)
        true_trajectory = y_val[start_idx:start_idx+max_seq_length-1]
        true_values_traj = [processor.cluster_centers[cluster] for cluster in true_trajectory]
        
        # Create initial sequences for multi-step-ahead predictions
        num_sequences = min(200, max_seq_length, length(X_val) - start_idx + 1)
        initial_sequences = [X_val[start_idx + i - 1] for i in 1:num_sequences]
        
        # Generate T=20 step predictions for all initial conditions
        generated_trajectories = []
        for initial_seq in initial_sequences
            traj = Int[]
            current_seq = copy(initial_seq)
            # Generate 20-step trajectory using deterministic (argmax) prediction
            for t in 1:20
                prob_output = model([current_seq])
                probs = softmax(prob_output, dims=1)[:, 1]
                next_cluster = argmax(probs)  # Deterministic prediction
                push!(traj, next_cluster)
                # Update sequence by removing first element and appending prediction
                current_seq = vcat(current_seq[2:end], next_cluster)
            end
            push!(generated_trajectories, traj)
        end
        
        # Analyze predictions at different delay horizons
        delays = [5, 10, 20]
        for (i, delay) in enumerate(delays)
            # Extract predictions at specific delay step
            delayed_values = [processor.cluster_centers[traj[delay]] for traj in generated_trajectories]
            
            # Get corresponding true values at the same time horizon
            max_compare_length = min(length(delayed_values), length(true_values_traj) - delay + 1)
            
            if max_compare_length > 0
                delayed_values_trimmed = delayed_values[1:max_compare_length]
                true_values_shifted = true_values_traj[delay:delay+max_compare_length-1]
                
                # Calculate prediction error statistics
                error = abs.(true_values_shifted - delayed_values_trimmed)
                mean_error = mean(error)
                
                # Create comparison plot for this delay horizon
                time_steps = 1:max_compare_length
                
                p = plot(time_steps, true_values_shifted, 
                         color=:black, linewidth=2, label="True (t+$delay)")
                plot!(p, time_steps, delayed_values_trimmed, 
                      color=:red, linewidth=2, alpha=0.7, label="Predicted (t→t+$delay)")
                plot!(p, title="Delay=$delay steps - Avg Error: $(round(mean_error, digits=3)) (Fixed Start)", 
                      xlabel="Time Steps", ylabel="Value", size=(800, 300))
                
                push!(plots_array, p)
            else
                # Handle edge case with insufficient data
                p = plot(title="Delay=$delay (insufficient data)", xlabel="Time Steps", ylabel="Value")
                push!(plots_array, p)
            end
        end
        
        # Row 6: Long-term trajectory statistical analysis from FIXED starting point
        if verbose
            println("Generating long trajectory for statistical analysis...")
        end
        
        # Use pre-determined fixed starting point for long trajectory
        start_idx_long = ANALYSIS_PARAMS["long_traj_start_idx"]
        initial_sequence_long = copy(X_val[start_idx_long])
        long_predicted_clusters = Int[]
        current_sequence_long = copy(initial_sequence_long)
        
        long_traj_seeds = ANALYSIS_PARAMS["long_traj_seeds"]
        
        # Generate 1000-step trajectory with fixed seeds for reproducibility
        for step in 1:1000
            # Use pre-determined seed for this step to ensure reproducibility
            Random.seed!(long_traj_seeds[step])
            prob_output = model([current_sequence_long])
            probs = softmax(prob_output, dims=1)[:, 1]
            next_cluster = argmax(probs)  # Use deterministic prediction
            push!(long_predicted_clusters, next_cluster)
            # Update sequence for next prediction
            current_sequence_long = vcat(current_sequence_long[2:end], next_cluster)
        end
        
        # Convert cluster indices to continuous values
        long_predicted_values = [processor.cluster_centers[cluster] for cluster in long_predicted_clusters]
        
        # Use original continuous data for statistical comparison baseline
        true_data_sample = y_data[1:min(1000, length(y_data))]
        
        # Left panel (row 6, col 1): Probability density function comparison
        try
            # Calculate kernel density estimates for smooth PDF comparison
            kde_predicted = kde(long_predicted_values)
            kde_true = kde(true_data_sample)
            
            p_pdf = plot(kde_true.x, kde_true.density, 
                         color=:black, linewidth=3, label="True PDF")
            plot!(p_pdf, kde_predicted.x, kde_predicted.density, 
                  color=:red, linewidth=3, alpha=0.7, label="Predicted PDF")
            plot!(p_pdf, title="PDF Comparison (Fixed Trajectory)", xlabel="Value", ylabel="Density")
            
            push!(plots_array, p_pdf)
            
        catch e
            # Fallback to histograms if KDE computation fails
            p_pdf = histogram(true_data_sample, bins=30, alpha=0.7, normalize=:pdf, 
                              color=:black, label="True PDF")
            histogram!(p_pdf, long_predicted_values, bins=30, alpha=0.5, normalize=:pdf, 
                       color=:red, label="Predicted PDF")
            plot!(p_pdf, title="PDF Comparison (Fixed Trajectory)", xlabel="Value", ylabel="Density")
            
            push!(plots_array, p_pdf)
        end
        
        # Right panel (row 6, col 2): Autocorrelation function comparison
        max_lag = 20
        acf_true = autocorrelation(true_data_sample, max_lag)
        acf_predicted = autocorrelation(long_predicted_values, max_lag)
        
        p_acf = plot(0:max_lag, acf_true, 
                     color=:black, linewidth=3, label="True ACF", marker=:circle, markersize=3)
        plot!(p_acf, 0:max_lag, acf_predicted, 
              color=:red, linewidth=3, alpha=0.7, label="Predicted ACF", marker=:square, markersize=3)
        plot!(p_acf, title="ACF Comparison (Fixed Trajectory)", xlabel="Lag", ylabel="Correlation")
        hline!(p_acf, [0], color=:gray, linestyle=:dash, alpha=0.5, label="")
        
        push!(plots_array, p_acf)
        
        # Create custom layout: 2x2 grid for ensemble plots, then 3 full-width delay plots, then 1x2 for statistics
        l = @layout [
            a b
            c d
            e{1.0w}
            f{1.0w}
            g{1.0w}
            h i
        ]
        
        # Combine all plots into comprehensive figure
        fig = plot(plots_array..., layout=l, size=(1200, 1800))
        
        # Save the comprehensive analysis plot
        savefig(fig, save_path)
        
        if verbose
            println("Comprehensive 6-row analysis completed successfully!")
            println("Analysis saved to: $save_path")
        end
        
        return fig
        
    catch e
        @warn "Analysis callback failed with error: $e"
        # Return empty plot to avoid breaking execution flow
        return plot(title="Analysis Failed: $e")
    end
end
