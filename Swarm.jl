# Various custom structs for FastAI to understand the swarm data
"""
    SwarmBlock{N,T}() <: Block

[`Block`](#) for a swarm of N agents, over T windows. `obs` is valid for `SwarmBlock{N,T}()`
if it is an N*3x-dimensional array with color or number element type.

"""


import FastAI.checkblock
import FastAI.mockblock
import FastAI.setup
import FastAI

# Assuming the dimentionality is 3 and x,y,z are unrolled
struct Swarm <: FastAI.Block
    # number of observed data points
    n::Int
    # window size
    w::Int
end

function checkblock(s::Swarm, obs::AbstractArray{T,N}) where {T<:Number,N<:Number} 
    size(obs) == (s.n, s.w)
end

function mockblock(s::Swarm)
    rand(Float32,s.n,s.w)
end

function setup(::Type{Swarm}, data)
    N,W = size(getobs(data, 1))
    Swarm(N,W)
end
