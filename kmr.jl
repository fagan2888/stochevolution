using QuantEcon
using Distributions
using PyPlot
using StatsBase


function kmr_markov_matrix_simultaneous(p::Float64, N::Int, epsilon::Float64)
    binom_pdf_1 = pdf(Binomial(N, epsilon/2))
    binom_pdf_2 = pdf(Binomial(N, 1-epsilon/2))
    binom_pdf_tie = pdf(Binomial(N, 1/2))

    P = Array(Float64, N+1, N+1)
    for n in 0:N
        P[n+1, :] = (n/N < p) * binom_pdf_1 +
                    (n/N == p) * binom_pdf_tie +
                    (n/N > p) * binom_pdf_2
    end
    return P
end


function kmr_markov_matrix_sequential(p::Float64, N::Int, epsilon::Float64)
    P = zeros(N+1, N+1)
    P[1, 1], P[1, 2] = 1 - epsilon * (1/2), epsilon * (1/2)
    @inbounds for n in 1:N-1
        P[n+1, n] = (n/N) * (
            epsilon * (1/2) +
            (1 - epsilon) * (((n-1)/(N-1) < p) + ((n-1)/(N-1) == p) * (1/2))
        )
        P[n+1, n+2] = ((N-n)/N) * (
            epsilon * (1/2) +
            (1 - epsilon) * ((n/(N-1) > p) + (n/(N-1) == p) * (1/2))
        )
        P[n+1, n+1] = 1 - P[n+1, n] - P[n+1, n+2]
    end
    P[end, end-1], P[end, end] = epsilon * (1/2), 1 - epsilon * (1/2)
    return P
end


function kmr_markov_chain(p::Float64, N::Int, epsilon::Float64,
                          revision::Symbol)
    if revision == :simultaneous
        P = kmr_markov_matrix_simultaneous(p, N, epsilon)
    elseif revision == :sequential
        P = kmr_markov_matrix_sequential(p, N, epsilon)
    else
        throw(ArgumentError)
    end
    state_values = 0:N
    return MarkovChain(P, state_values)
end


type KMR2x2
    p::Float64
    N::Int
    epsilon::Float64
    revision::Symbol
    mc::MarkovChain{Float64,Matrix{Float64},UnitRange{Int}}
end

function KMR2x2(p::Float64, N::Int, epsilon::Float64;
                revision::Symbol=:simultaneous)
    mc = kmr_markov_chain(p, N, epsilon, revision)
    return KMR2x2(p, N, epsilon, revision, mc)
end

function epsilon!(kmr::KMR2x2, epsilon)
    kmr.epsilon = epsilon
    kmr.mc = kmr_markov_chain(kmr.p, kmr.N, epsilon, kmr.revision)
    return kmr
end


QuantEcon.simulate(kmr::KMR2x2, ts_length::Int; init::Int=rand(0:kmr.N)) =
    simulate(kmr.mc, ts_length, init=init+1)

function simulate_cross_section(kmr::KMR2x2, ts_length::Int;
                                init::Vector{Int}=rand(0:kmr.N, num_reps),
                                num_reps::Int=1)
    X = Array(Int, ts_length, num_reps)
    simulate!(X, kmr.mc, init=init.+1)
    return X
end

function plot_sample_path(kmr::KMR2x2, x::Vector{Int};
                          ax=Union{}, show_plot=true)
    if show_plot
        fig, ax = subplots()
    end
    ax[:plot](x, alpha=0.5)
    ax[:set_ylim](0, kmr.N)
    ax[:set_title]("Sample path: \$\\varepsilon = $(kmr.epsilon)\$")
    ax[:set_xlabel]("Time")
    ax[:set_ylabel]("State")
    plt[:show]()
end

function plot_sample_path(kmr::KMR2x2, ts_length::Int; init::Int=rand(0:kmr.N),
                          ax=Union{}, show_plot=true)
    x = simulate(kmr, ts_length, init=init)
    plot_sample_path(kmr, x, ax=ax, show_plot=show_plot)
end


function plot_empirical_dist(kmr::KMR2x2, x::Vector{Int};
                             ax=Union{}, show_plot=true)
    h = fit(Histogram, x, 0:kmr.N+1, closed=:left)
    if show_plot
        fig, ax = subplots()
    end
    ax[:bar](0:kmr.N, h.weights, align="center")
    ax[:set_title]("Empirical distribution: \$\\varepsilon = $(kmr.epsilon)\$")
    ax[:set_xlim](-0.5, kmr.N+0.5)
    ax[:set_xlabel]("State")
    ax[:set_ylabel]("Frequency")
    plt[:show]()
end

function plot_empirical_dist(kmr::KMR2x2, ts_length::Int;
                             init::Int=rand(0:kmr.N),
                             ax=Union{}, show_plot=true)
    x = simulate(kmr, ts_length, init=init)
    plot_empirical_dist(kmr, x, ax=ax, show_plot=show_plot)
end


stationary_dist(kmr::KMR2x2) = stationary_distributions(kmr.mc)[1]

function plot_stationary_dist(kmr::KMR2x2; ax=Union{}, show_plot=true)
    if show_plot
        fig, ax = subplots()
    end
    ax[:bar](0:kmr.N, stationary_dist(kmr), align="center")
    ax[:set_title]("Stationary distribution: \$\\varepsilon = $(kmr.epsilon)\$")
    ax[:set_xlim](-0.5, kmr.N+0.5)
    ax[:set_xlabel]("State")
    ax[:set_ylabel]("Probability")
    plt[:show]()
end
