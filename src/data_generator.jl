using DifferentialEquations

"""
    Functions for generating synthetic dynamical systems data for testing and examples.
"""

"""
    lorenz63!(du, u, p, t)

The Lorenz system with parameters σ, ρ, β.
- u[1] = x, u[2] = y, u[3] = z
- p = [σ, ρ, β]
"""
function lorenz63!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    return nothing
end

"""
    generate_lorenz63_data(n_points; σ=10.0, ρ=28.0, β=8/3, dt=0.01, tspan=(0.0, 100.0), y_only=false, transient=nothing, return_time_steps=false)

Generate data from the Lorenz system with standard parameters.

# Arguments
- `n_points`: Number of data points to generate
- `σ`: Parameter σ (default: 10.0)
- `ρ`: Parameter ρ (default: 28.0)
- `β`: Parameter β (default: 8/3)
- `dt`: Time step for sampling (default: 0.01)
- `tspan`: Time span for integration (default: (0.0, 100.0))
- `y_only`: If true, returns only the y-coordinate (default: false)
- `transient`: Number of points to remove as transient. If nothing, removes 20% (default: nothing)
- `return_time_steps`: If true, returns a tuple (data, time_steps), otherwise just returns data (default: false)

# Returns
- If return_time_steps=false (default): time series data as matrix (or vector if y_only=true)
- If return_time_steps=true: tuple of (data, time_steps)
"""
function generate_lorenz63_data(n_points; 
                               σ=10.0, 
                               ρ=28.0, 
                               β=8/3, 
                               dt=0.01, 
                               tspan=(0.0, 100.0), 
                               y_only=false,
                               transient=nothing,
                               return_time_steps=false)
    # Parameters and initial conditions
    p = [σ, ρ, β]
    u0 = [1.0, 0.0, 0.0]
    
    # Solve the ODE
    prob = ODEProblem(lorenz63!, u0, tspan, p)
    sol = solve(prob, saveat=dt)
    
    # Extract solution to array and remove initial transient
    sol_array = Array(sol)
    transient_points = isnothing(transient) ? Int(round(0.2 * size(sol_array, 2))) : transient  # Remove first 20% as transient or specified amount
    
    # Sample n_points evenly from the remaining time series
    remaining_points = size(sol_array, 2) - transient_points
    if n_points >= remaining_points
        data = Matrix(sol_array[:, transient_points+1:end]')
        time_steps = sol.t[transient_points+1:end]
    else
        indices = transient_points .+ round.(Int, range(1, remaining_points, length=n_points))
        data = Matrix(sol_array[:, indices]')
        time_steps = sol.t[indices]
    end
    
    # Return data in the requested format
    if y_only
        result = data[:, 2]
    else
        result = data
    end
    
    return return_time_steps ? (result, time_steps) : result
end
