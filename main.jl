using FastAI, StaticArrays, Colors
using FastAI: FluxTraining, Image
using CSV
using DataFrames
using MLDataPattern
using FastAI: encodedblockfilled, decodedblockfilled
using Flux
import CairoMakie

df = DataFrame(CSV.File("data/data.csv"; header=false, types=Float32))

function normalise(M)
    min = minimum(minimum(eachcol(M)))
    max = maximum(maximum(eachcol(M)))
    return (M .- min) ./ (max - min)
end

normalised = Array(df) |> normalise

window_size = 60

data = slidingwindow(normalised', window_size, stride=1)

train_set, validate_set, test_set = splitobs(map(transpose, data), (0.7, 0.2));

getobs(train_set, 1)

function EmbeddingTask(block, encodings)
    sample = block
    encodedsample = x = y = ŷ = sample
    blocks = (; sample, x, y, ŷ, encodedsample)
    BlockTask(blocks, encodings)
end

task = EmbeddingTask(
    Swarm(900, 60),
    (ImagePreprocessing(),),
)

x = encodesample(task, Training(), getobs(train_set, 1))


BATCHSIZE = 48
dataloader = DataLoader(taskdataset(shuffleobs(train_set), task, Training()), BATCHSIZE)
dataiter = collect(dataloader)
for xs in dataiter
    print(size(xs))
    break
end

struct VAE{E,D}
    encoder::E
    decoder::D
end

Flux.@functor VAE

function (vae::VAE)(xs)
    μ, logσ² = vae.encoder(xs)
    zs = sample_latent(μ, logσ²)
    x̄s = vae.decoder(zs)
    return x̄s, (; μ, logσ²)
end


using Random: randn!
using Statistics: mean

sample_latent(μ::AbstractArray{T}, logσ²::AbstractArray{T}) where {T} =
    μ .+ exp.(logσ² ./ 2) .* randn!(similar(logσ²))

function βELBO(x, x̄, μ, logσ²; β=1)
    reconstruction_error = mean(sum(@.((x̄ - x)^2); dims=1))
    # D(N(μ, Σ)||N(0, I)) = 1/2 * (μᵀμ + tr(Σ) - length(μ) - log(|Σ|))
    kl_divergence = mean(sum(@.((μ^2 + exp(logσ²) - 1 - logσ²) / 2); dims=1))

    return reconstruction_error + β * kl_divergence
end

encoder =
    Chain(
        Conv((9,), 900 => 9000, relu; pad=SamePad()),
        MaxPool((2,)),
        Conv((5,), 9000 => 4500, relu; pad=SamePad()),
        MaxPool((2,)),
        Conv((5,), 4500 => 2250, relu; pad=SamePad()),
        MaxPool((3,)),
        Conv((3,), 2250 => 1000, relu; pad=SamePad()),
        Conv((3,), 1000 => 100, relu; pad=SamePad()),
        Flux.flatten,
        Parallel(
            tuple,
            Dense(500, 100), # μ
            Dense(500, 100), # logσ²
        ),
    ) |> gpu

decoder = Chain(
    Dense(100, 500, relu),
    (x -> reshape(x, 5, 100, :)),
    # 5x100xb
    ConvTranspose((3,), 100 => 1000, relu; pad=SamePad()),
    ConvTranspose((3,), 1000 => 2250, relu; pad=SamePad()),
    Upsample((3,)),
    # 15x2250xb
    ConvTranspose((5,), 2250 => 4500, relu; pad=SamePad()),
    Upsample((2,)),
    # 30x4500xb
    ConvTranspose((5,), 4500 => 9000, relu; pad=SamePad()),
    Upsample((2,)),
    # 60x9000xb
    ConvTranspose((9,), 9000 => 900; pad=SamePad()),
    # 60x900xb
) |> gpu

model = VAE(encoder, decoder)

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
    FluxTraining.runstep(learner, phase, (xs=batch,)) do handle, state
        gs = gradient(learner.params) do
            μ, logσ² = learner.model.encoder(state.xs)
            state.zs = sample_latent(μ, logσ²)
            state.x̄s = learner.model.decoder(state.zs)

            handle(FluxTraining.LossBegin())
            state.loss = learner.lossfn(Flux.flatten(state.xs), Flux.flatten(state.x̄s), μ, logσ²)

            handle(FluxTraining.BackwardBegin())
            return state.loss
        end
        handle(FluxTraining.BackwardEnd())
        Flux.Optimise.update!(learner.optimizer, learner.params, gs)
    end
end

function FluxTraining.on(
    ::FluxTraining.StepBegin,
    ::VAETrainingPhase,
    cb::ToDevice,
    learner,
)
    learner.step.xs = cb.movedatafn(learner.step.xs)
end

learner = Learner(model, (), ADAM(), βELBO, ToGPU())

FluxTraining.removecallback!(learner, ProgressPrinter);

fitonecycle!(
    learner,
    2,
    0.01;
    phases=(VAETrainingPhase() => dataiter,)
)