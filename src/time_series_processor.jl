# Import required packages
using Clustering
using Statistics
using Random

"""
    TimeSeriesProcessor

Handles the preprocessing of time series data, including normalization and clustering.
"""
mutable struct TimeSeriesProcessor
    raw_data::Vector{Float64}        # Original time series data
    normalized_data::Vector{Float64} # Normalized time series data
    num_clusters::Int                # Number of clusters for k-means
    kmeans_model                     # K-means clustering model
    cluster_centers::Vector{Float64} # Centers of the clusters in ORIGINAL scale
    cluster_assignments::Vector{Int} # Cluster assignments for each time point
    data_mean::Float64              # Mean of original data for denormalization
    data_std::Float64               # Standard deviation of original data for denormalization
    
    """
        TimeSeriesProcessor(data, num_clusters)
        TimeSeriesProcessor(data; num_clusters)

    Initialize a TimeSeriesProcessor with time series data and number of clusters.

    # Arguments
    - `data`: Vector or Matrix of time series data points. If matrix, first column is used.
    - `num_clusters`: Number of clusters to use for k-means
    """
    function TimeSeriesProcessor(data::Union{Vector{Float64}, Matrix{Float64}}, num_clusters::Int)
        # If data is a matrix, take the first column
        time_series = data isa Matrix ? vec(data[:, 1]) : data
        
        # Store normalization parameters
        data_mean = mean(time_series)
        data_std = std(time_series)
        
        # Normalize the data to have zero mean and unit variance
        normalized_data = (time_series .- data_mean) ./ data_std
        
        # Initialize with empty clustering results
        new(time_series, normalized_data, num_clusters, nothing, Float64[], Int[], data_mean, data_std)
    end

    # Constructor with keyword arguments
    function TimeSeriesProcessor(data::Union{Vector{Float64}, Matrix{Float64}}; num_clusters::Int)
        TimeSeriesProcessor(data, num_clusters)
    end
end

"""
    process(processor::TimeSeriesProcessor)

Process the time series data: normalize and perform k-means clustering with improved numerical stability.

# Returns
- Vector of cluster assignments for each time point
"""
function process(processor::TimeSeriesProcessor)
    # Normalize data more robustly using median absolute deviation
    X = reshape(processor.normalized_data, length(processor.normalized_data), 1)
    
    # Add small random noise to ensure numerical stability
    # The noise is proportional to the data scale
    data_scale = std(X)
    epsilon = Float64(1e-6)
    noise_scale = max(data_scale * epsilon, Float64(1e-8))
    X = X .+ noise_scale .* randn(size(X))
    
    # Ensure data is in correct format and handle potential NaN/Inf
    X = replace(X, NaN => 0.0, Inf => prevfloat(Inf), -Inf => nextfloat(-Inf))
    
    # The Clustering.jl package expects observations as columns
    X_transposed = transpose(X)
    
    # Determine effective number of clusters based on data variability
    variance = var(X)
    min_variance_threshold = Float64(1e-10)
    
    if variance < min_variance_threshold
        @warn "Very low data variance detected, adding stabilizing noise"
        X_transposed = X_transposed .+ Float64(1e-5) .* randn(size(X_transposed))
    end
    
    # Determine effective number of clusters
    effective_clusters = if variance < min_variance_threshold
        2  # Use minimum number of clusters for nearly constant data
    else
        # Estimate unique points using a tolerance-based approach
        tolerance = sqrt(eps(Float64))  # Use square root of machine epsilon as tolerance
        points_list = [X_transposed[:, i] for i in 1:size(X_transposed, 2)]
        
        # Count effectively unique points
        is_unique = falses(length(points_list))
        is_unique[1] = true
        unique_count = 1
        
        for i in 2:length(points_list)
            if !any(j -> j < i && norm(points_list[i] - points_list[j]) < tolerance, 1:i-1)
                is_unique[i] = true
                unique_count += 1
            end
        end
        
        min(processor.num_clusters, max(2, unique_count))
    end
    
    @info "Using $effective_clusters clusters (requested: $(processor.num_clusters))"
    
    # Perform k-means clustering with multiple restarts
    best_kmeans = nothing
    best_objective = Inf
    n_restarts = 5
    
    for i in 1:n_restarts
        try
            # Run k-means with current initialization
            kmeans_result = kmeans(X_transposed, effective_clusters;
                                 maxiter=300,
                                 tol=Float64(1e-6),
                                 display=:none)
            
            # Update if this is the best result so far
            if kmeans_result.totalcost < best_objective
                best_kmeans = kmeans_result
                best_objective = kmeans_result.totalcost
            end
        catch e
            @warn "K-means attempt $i failed" exception=e
            continue
        end
    end
    
    if isnothing(best_kmeans)
        error("Failed to perform clustering after $n_restarts attempts")
    end
    
    # Store results
    processor.kmeans_model = best_kmeans
    
    # Convert cluster centers back to original scale
    normalized_centers = best_kmeans.centers'  # This gives us centers in normalized space
    original_centers = vec(normalized_centers) .* processor.data_std .+ processor.data_mean
    processor.cluster_centers = original_centers
    
    processor.cluster_assignments = assignments(best_kmeans)
    
    # Validate results
    if any(isnan, processor.cluster_centers) || any(isnan, processor.cluster_assignments)
        error("NaN values detected in clustering results")
    end
    
    return processor.cluster_assignments
end

"""
    get_sequence_dataset(processor::TimeSeriesProcessor, sequence_length::Int)

Process the time series data into sequences for training.

# Arguments
- `processor`: TimeSeriesProcessor instance
- `sequence_length`: Length of sequences to create

# Returns
- Tuple of (X, y) where X is a vector of sequences and y is a vector of next values
"""
function get_sequence_dataset(processor::TimeSeriesProcessor, sequence_length::Int)
    # Ensure clustering has been performed
    if isempty(processor.cluster_assignments) || isnothing(processor.kmeans_model)
        @info "Running clustering before getting sequences..."
        process(processor)
    end
    
    # Get cluster assignments
    clusters = processor.cluster_assignments
    
    # Create sequences and targets
    X = Vector{Int}[]
    y = Int[]
    
    # Iterate through the data to create sequences
    for i in 1:(length(clusters) - sequence_length)
        sequence = clusters[i:(i + sequence_length - 1)]
        next_value = clusters[i + sequence_length]
        push!(X, sequence)
        push!(y, next_value)
    end
    
    return X, y
end

"""
    create_minibatches(X, y, batch_size)

Create minibatches from sequence dataset.

# Arguments
- `X`: Input sequences
- `y`: Target outputs
- `batch_size`: Size of each minibatch

# Returns
- Vector of (batch_X, batch_y) tuples
"""
function create_minibatches(X, y, batch_size)
    n = length(X)
    
    # Use Random.shuffle instead of shuffle for more explicit control
    indices = Random.shuffle(1:n)
    
    num_batches = ceil(Int, n / batch_size)
    
    # Pre-allocate batches array with specific type annotation
    # This is more efficient and Zygote-friendly
    batches = Vector{Tuple{Vector{Vector{Int}}, Vector{Int}}}(undef, num_batches)
    
    for i in 1:num_batches
        start_idx = (i-1)*batch_size+1
        end_idx = min(i*batch_size, n)
        batch_indices = indices[start_idx:end_idx]
        
        # Use non-mutating approach to create batches
        batch_X = [X[j] for j in batch_indices]
        batch_y = [y[j] for j in batch_indices]
        
        # Assign to pre-allocated array - safer than push!
        batches[i] = (batch_X, batch_y)
    end
    
    return batches
end
