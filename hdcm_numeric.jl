# numerical solution for 2cpt

using Pkg;
Pkg.activate(".");
using OrdinaryDiffEq
using CSV, DataFramesMeta, Chain
using Plots, CairoMakie, AlgebraOfGraphics
using Flux, DiffEqFlux
using Random
using StatsBase
using CategoricalArrays
using Turing, Distributions, LinearAlgebra, StatsPlots, ReverseDiff
using Optim
using MLJ, MLJFlux, ShapML
using Distributed
using JLD2

# setup directories
resDir = joinpath("results", "hdcm_numeric", "analysis3")
mkpath(resDir)

addprocs(4)

rng = Random.default_rng()
Random.seed!(12345)

# Use reverse_diff due to the number of parameters in neural networks.
Turing.setadbackend(:reversediff)

#########################

## analysis3 ##
# DA
dat1 = CSV.read("data/analysis3.csv", DataFrame)
dat2 = @rsubset(dat1, :STUDY <= "101-DEMO-001", :ID <= 30, :EVID == 0)  # single dose study; will train against the first 10 subjects


# extract info
ids = unique(dat2.ID)
ntrain = 10
ids_train = sort(sample(1:30, 10, replace=false))
dat_train = @rsubset(dat2, :ID in ids_train)

dat_dose = @rsubset(dat1[dat1.ID.<=30, :], :EVID == 1)
dat_dose.AMT = parse.(Float64, dat_dose.AMT)
dat_dose.AMT = dat_dose.AMT .* 1e3  # get right units 
dat_train.DV[dat_train.DV.<=0.0] .= 1e-12  # avoid 0
nSubj = length(unique(dat2.ID))
nSubj_train = length(unique(dat_train.ID))
doses = dat_dose.AMT
blq = dat2.BLQ
blq_train = dat_train.BLQ
L = 10.0  # assuming loq is 10

times = []
for i in ids
    times_ = dat2.TIME[dat2.ID.==i]
    push!(times, times_)
end

# get covs and other stuff
contcovs = dat_dose[:, ["AGE", "WT", "ALB", "EGFR"]]
catcovs = dat_dose[:, "SEX"]

# unnormalised covs
#covs = Array(dat_dose[:,["AGE","WT","ALB","EGFR","SEX"]])
covs = Array(dat_dose[:, ["AGE", "WT", "ALB", "EGFR"]])

# normalized covs
dt = StatsBase.fit(UnitRangeTransform, Array(contcovs), dims=1)
covsn = StatsBase.transform(dt, Array(contcovs))
#covsn = hcat(covsn, catcovs)

####

## some EDA
# plot data
pl = AlgebraOfGraphics.data(dat2[dat2.BLQ .== 0,:])
l1 = mapping(:TIME => "Time (h)", :DV => "Concentration (ng/mL)", group=:ID => nonnumeric, layout=:DOSE => nonnumeric) * visual(ScatterLines)
pl_databyDose = draw(pl * (l1), facet=(; linkyaxes=:none))
pl_databyDose_log = draw(pl * (l1), facet=(; linkyaxes=:none), axis=(yscale=log10,))

l2 = mapping(:TIME => "Time (h)", :DV => "Concentration (ng/mL)", group=:ID => nonnumeric, color=:DOSE => nonnumeric => "Dose (mg)") * visual(ScatterLines)
pl_data = draw(pl * (l2), facet=(; linkyaxes=:none))
pl_data_log = draw(pl * (l2), facet=(; linkyaxes=:none), axis=(yscale=log10,))

## table
#dat2_summ = describe(select(dat2, Not(:C)))
dat2_summ = describe(select(dat2, :ID, :DV, :BLQ, :AGE, :SEX, :WT, :HT, :EGFR, :ALB, :BMI, :AAG, :SCR, :AST, :ALT))
#describe(dat2_summ)

#dat2_summ_obs = DataFrame(Observations = nrow(dat2), BLQ = sum(blq))
dat2_summ.Observations .= nrow(dat2)
dat2_summ.BLQ .= sum(blq)
@rtransform!(dat2_summ, :range = string(:min, ", ", :max))
CSV.write(joinpath(resDir, "df_data_summ.csv"), dat2_summ)

####

# model
function pk2cpt!(du, u, p, t)
    ka, CL, V2, Q, V3 = p
    depot, cent, peri = u

    du[1] = ddepot = -ka * depot
    du[2] = dcent = ka * depot / V2 - (CL / V2) * cent - (Q / V2) * cent + (Q / V3) * peri
    du[3] = dperi = (Q / V3) * cent - (Q / V3) * peri
end

p = [1.5, 3.5, 60.0, 4.0, 70.0]
u0 = [5000.0, 0.0, 0.0]
tspan = (0.0, times[1][end])

# define problem
prob = ODEProblem(pk2cpt!, u0, tspan, p)

sol = solve(prob, Tsit5(), saveat=times[1])

Plots.plot(sol.t, sol[2, :])
Plots.scatter!(times[1], dat2.DV[dat2.ID.==1])

####

# get MLE/MAP of PK params
## Bayesian inference ##
@model function bayes_pk(data, prob, doses, times, blq, nSubj)
    # Create the weight and bias vector.
    σ ~ truncated(Cauchy(0, 0.5), 0.0, 2.0)  # residual error

    ka ~ LogNormal(log(1), 0.2)
    CL ~ LogNormal(log(5), 0.2)
    V2 ~ LogNormal(log(50), 0.2)
    Q ~ LogNormal(log(5), 0.2)
    V3 ~ LogNormal(log(50), 0.2)

    p_ = [ka, CL, V2, Q, V3]

    # function to update ODE problem with newly sampled params
    function prob_func(prob, i, repeat)
        u0_ = [doses[i], 0.0, 0.0]
        times_ = times[i]
        tspan_ = (0.0, times_[end])
        remake(prob, p=p_, u0=u0_, tspan=tspan_, saveat=times_)
    end

    # define an ensemble problem and simulate the population
    tmp_ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    # solving can be parallelized by replacing EnsembleSerial() with EnsembleThreads() but there is no need since chains will be parallelized later
    # and having nested parallelization might lead to non reproducible results
    tmp_ensemble_sol = solve(tmp_ensemble_prob, Tsit5(), EnsembleSerial(), trajectories=nSubj)

    preds = reduce(vcat, [tmp_ensemble_sol[i][2, :] for i in 1:nSubj])

    preds[preds.<=0] .= 1e-12

    for i in eachindex(preds)
        if blq[i] == 0
            #Turing.@addlogprob! logpdf(Normal(preds[i], preds[i] * σ), data[i])
            data[i] ~ Normal(preds[i], preds[i] * σ)
        else  # censored data have likelihood (probablity) of p(y <= L) where L is a lower bound; this is equal to the normal CDF; note: if right censored data p(y >= U) = 1 - normal CDF (logccdf)
            Turing.@addlogprob! logcdf(Normal(preds[i], preds[i] * σ), L)
        end
    end
end;

# instantiate model
mod = bayes_pk(dat2.DV, prob, doses, times, blq, nSubj);

# MLE
#p_mle = Optim.optimize(mod, MLE(), NelderMead())
# get standard error; sqrt of diagonal of asymptotic varince-covariance matrix
##StatsBase.coeftable(p_mle)  # gives error
#varcov = vcov(p_mle)
#err = sqrt.(diag(varcov))  
#p_mle_array = p_mle.values.array

#df_mle = DataFrame(value = p_mle_array, err=err)

# MAP
p_map = Optim.optimize(mod, MAP(), NelderMead())
#StatsBase.coeftable(p_map)
#varcov = vcov(p_map)
#err = sqrt.(diag(varcov)) 
p_map_array = p_map.values.array

####

ncovs = length(covsn[1, :])

# with initialization from MLE/MAP
NN = Flux.Chain(
    Flux.Dense(ncovs, 6, Flux.swish), # hidden layer
    #Flux.Dense(6, 5), x -> abs.(x ./ mean(x, dims=1)) .+ p
    Flux.Dense(6, 5), x -> (Flux.celu.(x, 0.999) .+ ones(5)) .* p_map_array[2:end]
)

# Extract weights and a helper function to reconstruct NN from weights
pinit, re = Flux.destructure(NN)
NN(covsn[1, :])  # sanity check
nwts = length(pinit) # number of paraemters in NN

# define an ensemble problem and simulate the population
# function to update ODE problem with newly sampled params
function prob_func(prob, i, repeat)
    u0_ = [doses[i], 0.0, 0.0]
    times_ = times[i]
    tspan_ = (0.0, times_[end])
    p_ = NN(covsn[i, :])
    remake(prob, p=p_, u0=u0_, tspan=tspan_, saveat=times_)
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
# solving can be parallelized by replacing EnsembleSerial() with EnsembleThreads() but there is no need since chains will be parallelized later
# and having nested parallelization might lead to non reproducible results
ensemble_sol = solve(ensemble_prob, Tsit5(), EnsembleThreads(), trajectories=nSubj)

Plots.plot(ensemble_sol, idxs=2)

###############

## Bayesian inference ##
@model function bayes_nn(data, prob, blq, nSubj, doses, times, covs, nwts, re; α=0.09)
    # Create the weight and bias vector.
    σ ~ truncated(Cauchy(0, 0.5), 0.0, 2.0)  # residual error
    w ~ filldist(Normal(0.0, α), nwts) # NN weights; alpha/standard deviation 0.6 = 60% RSD
    #w ~ MvNormal(zeros(nwts), I * α^2)  # NN weights; alpha/variance 0.09 = 30% RSD

    # Construct NN from weights
    NN_ = re(w)

    # IIV; non-centered
    ω_CL ~ truncated(Cauchy(0, 0.5), 0.0, 1.0)  # intersubject variability SD
    η_CL ~ filldist(Normal(0.0, 1.0), nSubj)  # individual ηs for random effects

    ω_V2 ~ truncated(Cauchy(0, 0.5), 0.0, 1.0)  # intersubject variability SD
    η_V2 ~ filldist(Normal(0.0, 1.0), nSubj)  # individual ηs for random effects

    ω_ka ~ truncated(Cauchy(0, 0.5), 0.0, 1.0)  # intersubject variability SD
    η_ka ~ filldist(Normal(0.0, 1.0), nSubj)  # individual ηs for random effects

    # function to update ODE problem with newly sampled params
    function prob_func(prob, i, repeat)
        u0_ = [doses[i], 0.0, 0.0]
        times_ = times[i]
        tspan_ = (0.0, times_[end])
        p_ = NN_(covs[i, :])
        ka, CL, V2, Q, V3 = p_
        # individual params ; 
        CLᵢ = CL .* exp.(ω_CL .* η_CL[i])  # non-centered parameterization
        V2ᵢ = V2 .* exp.(ω_V2 .* η_V2[i])  # non-centered parameterization
        kaᵢ = ka .* exp.(ω_ka .* η_ka[i])  # non-centered parameterization
        ps_ = [kaᵢ, CLᵢ, V2ᵢ, Q, V3]
        remake(prob, p=ps_, u0=u0_, tspan=tspan_, saveat=times_)
    end

    # define an ensemble problem and simulate the population
    tmp_ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
    # solving can be parallelized by replacing EnsembleSerial() with EnsembleThreads() but there is no need since chains will be parallelized later
    # and having nested parallelization might lead to non reproducible results
    tmp_ensemble_sol = solve(tmp_ensemble_prob, Tsit5(), EnsembleSerial(), trajectories=nSubj)

    preds = reduce(vcat, [tmp_ensemble_sol[i][2, :] for i in 1:nSubj])

    preds[preds.<=0] .= 1e-12

    for i in eachindex(preds)
        if blq[i] == 0
            #Turing.@addlogprob! logpdf(Normal(preds[i], preds[i] * σ), data[i])
            data[i] ~ Normal(preds[i], preds[i] * σ)
        else  # censored data have likelihood (probablity) of p(y <= L) where L is a lower bound; this is equal to the normal CDF; note: if right censored data p(y >= U) = 1 - normal CDF (logccdf)
            Turing.@addlogprob! logcdf(Normal(preds[i], preds[i] * σ), L)
        end
    end
end;

# define the Bayesian model object
mod = bayes_nn(dat_train.DV, prob, blq_train, nSubj_train, doses[ids_train], times[ids_train], covsn[ids_train, :], nwts, re; α=0.6);

# sample 
## sampling conditions
nburn = 500  #250
nsampl = 500  #250
nchains = 4
adapt_delta = 0.8  #0.65

## sampling
#@time ch = sample(mod, HMC(0.1,10), MCMCSerial(), nsampl, nchains)
#@time ch = sample(mod, NUTS(nburn, adapt_delta), MCMCSerial(), nsampl, nchains)
#@time ch_serial = sample(mod, NUTS(nburn,adapt_delta), MCMCSerial(), nsampl, nchains)
@time ch = sample(mod, NUTS(nburn, adapt_delta), MCMCThreads(), nsampl, nchains)
#@time ch_prior = sample(mod, Prior(), MCMCSerial(), nsampl, nchains)  # serial
@time ch_prior = sample(mod, Prior(), MCMCThreads(), nsampl, nchains)  # serial

## save mcmcchains; uncomment to save generated chains; caution: this will overwrite the already saved chains
JLD2.save(joinpath(resDir, "chains_numeric_analysis3.jld2"), "chains", ch, "chains_prior",ch_prior)
#ch = JLD2.load(joinpath(resDir, "chains_numeric_analysis3.jld2"), "chains")
#ch_prior = JLD2.load(joinpath(resDir, "chains_numeric_analysis3.jld2"), "chains_prior")

###############

## post-processing
df_chain = DataFrame(Array(ch), :auto)
df_chain_prior = DataFrame(Array(ch_prior), :auto)


## diagnostics ##
# Stats
summ, quant = describe(ch)
summ_prior, quant_prior = describe(ch_prior)

# save
df_summ = DataFrame(summ)
df_quant = DataFrame(quant)
df_summ_prior = DataFrame(summ_prior)
df_quant_prior = DataFrame(quant_prior)

df_summquant = @transform(hcat(df_summ, select(df_quant, Not(:parameters))), :distribution = "posterior")
df_summquant_prior = @transform(hcat(df_summ_prior, select(df_quant_prior, Not(:parameters))), :distribution = "prior")

df_summquant_all = vcat(df_summquant, df_summquant_prior)

# save
CSV.write(joinpath(resDir, "df_summquant_all.csv"), df_summquant_all)

## create dataframe for summary of summary
df_summquant_select = @chain df_summquant_all begin
    @rtransform(:Mean = string(round(:mean; sigdigits=3), " (", round(:std; sigdigits=3), ")"),
        :Median = string(round(:"50.0%"; sigdigits=3), " (", round(:"2.5%"; sigdigits=3), ", ", round(:"97.5%"; sigdigits=3), ")"))
    select(:parameters => :Parameter, :Mean => "Mean (SD)", :Median => "Median (95% CI)", :rhat, :ess_bulk, :ess_tail)
end

# save
CSV.write(joinpath(resDir, "df_summquant_select.csv"), df_summquant_select)

# plots
## trace plots ; we will split plots and join later for better view
pl_chains = StatsPlots.plot(ch[:, 1:10, :])
pl_chains_prior = StatsPlots.plot(ch_prior[:, 1:10, :])

pl_chains_sample = StatsPlots.plot(ch[:, [1, 8, 67, 78, 89], :])
pl_chains_sample_prior = StatsPlots.plot(ch_prior[:, [1, 8, 67, 78, 89], :])

pl_density_sample = StatsPlots.density(ch[:, [1, 8, 67, 78, 89], :])
pl_density_sample_prior = StatsPlots.density(ch_prior[:, [1, 8, 67, 78, 89], :])

##############

# summarise parameters #

# load chains
#ch = JLD2.load("results/hdcm_numeric/analysis3/results.jld2", "chains")
#ch_prior = JLD2.load("results/hdcm_numeric/analysis3/results.jld2", "chains_prior")

w_prior = MCMCChains.group(ch_prior, :w).value;
η_ka_prior = MCMCChains.group(ch_prior, :η_ka).value;
η_CL_prior = MCMCChains.group(ch_prior, :η_CL).value;
η_V2_prior = MCMCChains.group(ch_prior, :η_V2).value;
σ_prior = MCMCChains.group(ch_prior, :σ).value;
ω_ka_prior = MCMCChains.group(ch_prior, :ω_ka).value;
ω_CL_prior = MCMCChains.group(ch_prior, :ω_CL).value;
ω_V2_prior = MCMCChains.group(ch_prior, :ω_V2).value;

w = MCMCChains.group(ch, :w).value;
η_ka = MCMCChains.group(ch, :η_ka).value;
η_CL = MCMCChains.group(ch, :η_CL).value;
η_V2 = MCMCChains.group(ch, :η_V2).value;
σ = MCMCChains.group(ch, :σ).value;
ω_ka = MCMCChains.group(ch, :ω_ka).value;
ω_CL = MCMCChains.group(ch, :ω_CL).value;
ω_V2 = MCMCChains.group(ch, :ω_V2).value;

# get pk pars
pars_pop = Array{Float64}(undef, nsampl, length(p), nchains, nSubj_train)
pars_ind = Array{Float64}(undef, nsampl, length(p), nchains, nSubj_train)

for i in 1:nsampl
    for j in 1:nSubj_train
        for k in 1:nchains
            NN_ = re(w[i, :, k])
            pars_pop[i, :, k, j] = NN_(covsn[ids_train, :][j, :])
            # add random effects
            pars_ind[i, :, k, j] = NN_(covsn[ids_train, :][j, :])
            pars_ind[i, 1, k, j] = pars_ind[i, 1, k, j] * exp.(ω_ka[i, 1, k] .* η_ka[i, j, k])
            pars_ind[i, 2, k, j] = pars_ind[i, 2, k, j] * exp.(ω_CL[i, 1, k] .* η_CL[i, j, k])
            pars_ind[i, 3, k, j] = pars_ind[i, 3, k, j] * exp.(ω_V2[i, 1, k] .* η_V2[i, j, k])
            #push!(pars, NN_(covsn[j,:]))
        end
    end
end

pars_pop_summ = Array{Float64}(undef, nSubj_train, length(p))
pars_ind_summ = Array{Float64}(undef, nSubj_train, length(p))

for i in 1:nSubj_train
    pars_pop_tmp = []
    pars_ind_tmp = []
    for j in 1:nchains
        push!(pars_pop_tmp, pars_pop[:, :, j, i])
        push!(pars_ind_tmp, pars_ind[:, :, j, i])
    end
    pars_pop_tmp2 = vcat(pars_pop_tmp...)
    pars_ind_tmp2 = vcat(pars_ind_tmp...)

    pars_pop_summ[i, :] = mean(pars_pop_tmp2, dims=1)
    pars_ind_summ[i, :] = mean(pars_ind_tmp2, dims=1)
end

# concatenate
#pars_pop2 = vcat([vcat(pars_pop[:,:,:,i]) for i in 1:nSubj_train]...)
pars_pop2 = vcat([vcat(pars_pop[:, :, i, :]) for i in 1:nchains]...)
pars_ind2 = vcat([vcat(pars_ind[:, :, i, :]) for i in 1:nchains]...)

# create DF
params = ["ka", "CL", "V2", "Q", "V3"]
df_pars_pop = DataFrame()
df_pars_ind = DataFrame()
for i in 1:nSubj_train
    df_tmp_pop = DataFrame(pars_pop2[:, :, i], params)
    df_tmp_ind = DataFrame(pars_ind2[:, :, i], params)

    @transform!(df_tmp_pop, :ID = i, :sample = 1:nsampl*nchains)
    @transform!(df_tmp_ind, :ID = i, :sample = 1:nsampl*nchains)

    append!(df_pars_pop, df_tmp_pop)
    append!(df_pars_ind, df_tmp_ind)
end

df_pars_pop2 = sort(stack(df_pars_pop, Not([:ID, :sample]), variable_name=:parameter), [:sample, :ID])
df_pars_ind2 = sort(stack(df_pars_ind, Not([:ID, :sample]), variable_name=:parameter), [:sample, :ID])

# summarise

# oroginal values ;; https://ghe.metrumrg.com/example-projects/bbr-nonmem-poppk-foce/blob/main/model/pk/106/106.ctl
df_orig = DataFrame(parameter = ["ka","CL","V2","Q","V3"],
                    true_value = [1.5, 3.5, 60.0, 4.0, 70.0])

df_pars_pop3 = @chain df_pars_pop2 begin
    groupby([:sample, :parameter])
    @transform(:avg = mean(:value),
        :sd = std(:value),
        :med = quantile(:value, 0.5),
        :lo = quantile(:value, 0.05),
        :hi = quantile(:value, 0.95))

    groupby(:parameter)
    @transform(:medAvg = quantile(:avg, 0.5),
        :loAvg = quantile(:avg, 0.025),
        :hiAvg = quantile(:avg, 0.975),
        :medSD = quantile(:sd, 0.5),
        :loSD = quantile(:sd, 0.025),
        :hiSD = quantile(:sd, 0.975),
        :medMed = quantile(:med, 0.5),
        :loMed = quantile(:med, 0.025),
        :hiMed = quantile(:med, 0.975),
        :medLo = quantile(:lo, 0.5),
        :loLo = quantile(:lo, 0.025),
        :hiLo = quantile(:lo, 0.975),
        :medHi = quantile(:hi, 0.5),
        :loHi = quantile(:hi, 0.025),
        :hiHi = quantile(:hi, 0.975))
end

df_pars_ind3 = @chain df_pars_ind2 begin
    groupby([:sample, :parameter])
    @transform(:avg = mean(:value),
        :sd = std(:value),
        :med = quantile(:value, 0.5),
        :lo = quantile(:value, 0.05),
        :hi = quantile(:value, 0.95))

    groupby(:parameter)
    @transform(:medAvg = quantile(:avg, 0.5),
        :loAvg = quantile(:avg, 0.025),
        :hiAvg = quantile(:avg, 0.975),
        :medSD = quantile(:sd, 0.5),
        :loSD = quantile(:sd, 0.025),
        :hiSD = quantile(:sd, 0.975),
        :medMed = quantile(:med, 0.5),
        :loMed = quantile(:med, 0.025),
        :hiMed = quantile(:med, 0.975),
        :medLo = quantile(:lo, 0.5),
        :loLo = quantile(:lo, 0.025),
        :hiLo = quantile(:lo, 0.975),
        :medHi = quantile(:hi, 0.5),
        :loHi = quantile(:hi, 0.025),
        :hiHi = quantile(:hi, 0.975))
end

df_pars_pop_summ = combine(first, groupby(df_pars_pop3, [:parameter]))
df_pars_pop_summ2 = @chain df_pars_pop_summ begin
    # compare to original
    leftjoin(df_orig, on = :parameter)
    @rtransform(:Error = ((:medMed - :true_value)/:true_value) * 100.0)
    # add stats and reformat
    @rtransform(:Mean = string(round(:medAvg; sigdigits=3), " (", round(:loAvg; sigdigits=3), ", ", round(:hiAvg; sigdigits=3), ")"),
        :SD = string(round(:medSD; sigdigits=3), " (", round(:loSD; sigdigits=3), ", ", round(:hiSD; sigdigits=3), ")"),
        :Median = string(round(:medMed; sigdigits=3), " (", round(:loMed; sigdigits=3), ", ", round(:hiMed; sigdigits=3), ")"),
        :fifth = string(round(:medLo; sigdigits=3), " (", round(:loLo; sigdigits=3), ", ", round(:hiLo; sigdigits=3), ")"),
        :ninetyfifth = string(round(:medHi; sigdigits=3), " (", round(:loHi; sigdigits=3), ", ", round(:hiHi; sigdigits=3), ")"))
    select(:parameter => :Parameter, :Mean => "Mean (95% CI)", :SD => "SD (95% CI)", :Median => "Median (95% CI)", :fifth => "5th percentile (95% CI)", :ninetyfifth => "95th percentile (95% CI)", :Error => "Error (%)")
end
df_pars_pop_summ2."Error (%)" = round.(abs.(df_pars_pop_summ2."Error (%)"); sigdigits=3)

df_pars_ind_summ = combine(first, groupby(df_pars_ind3, [:parameter]))
df_pars_ind_summ2 = @chain df_pars_ind_summ begin
    # compare to original
    leftjoin(df_orig, on = :parameter)
    @rtransform(:Error = ((:medMed - :true_value)/:true_value) * 100.0)
    # add stats and reformat
    @rtransform(:Mean = string(round(:medAvg; sigdigits=3), " (", round(:loAvg; sigdigits=3), ", ", round(:hiAvg; sigdigits=3), ")"),
        :SD = string(round(:medSD; sigdigits=3), " (", round(:loSD; sigdigits=3), ", ", round(:hiSD; sigdigits=3), ")"),
        :Median = string(round(:medMed; sigdigits=3), " (", round(:loMed; sigdigits=3), ", ", round(:hiMed; sigdigits=3), ")"),
        :fifth = string(round(:medLo; sigdigits=3), " (", round(:loLo; sigdigits=3), ", ", round(:hiLo; sigdigits=3), ")"),
        :ninetyfifth = string(round(:medHi; sigdigits=3), " (", round(:loHi; sigdigits=3), ", ", round(:hiHi; sigdigits=3), ")"))
    select(:parameter => :Parameter, :Mean => "Mean (95% CI)", :SD => "SD (95% CI)", :Median => "Median (95% CI)", :fifth => "5th percentile (95% CI)", :ninetyfifth => "95th percentile (95% CI)", :Error => "Error (%)")
end
df_pars_ind_summ2."Error (%)" = round.(abs.(df_pars_ind_summ2."Error (%)"); sigdigits=3)


CSV.write(joinpath(resDir, "df_pars_pop_summ2.csv"), df_pars_pop_summ2)
CSV.write(joinpath(resDir, "df_pars_ind_summ2.csv"), df_pars_ind_summ2)


## recretae for publication

df_pars_pop_summ2_paper = select(df_pars_pop_summ2, [1,4,5,6,7])
df_pars_ind_summ2_paper = select(df_pars_ind_summ2, [1,4,5,6,7])

CSV.write(joinpath(resDir, "df_pars_pop_summ2_paper.csv"), df_pars_pop_summ2_paper)
CSV.write(joinpath(resDir, "df_pars_ind_summ2_paper.csv"), df_pars_ind_summ2_paper)


###############

## predictive checks ##
#--# conditional on chains #--#

# we will first create a vector of "missing" values to pass to the Bayesian fit function
# Turing will understand that the created model object is meant to simulate rather than fit
dat_missing = Vector{Missing}(missing, length(dat_train.DV)) # vector of `missing`
mod_pred = bayes_nn(dat_missing, prob, blq_train, nSubj_train, doses[ids_train], times[ids_train], covsn[ids_train, :], nwts, re; α=0.6);
pred = Turing.predict(mod_pred, ch; include_all=false)  # include_all = false means sampling new !!
pred_prior = Turing.predict(mod_pred, ch_prior; include_all=false)

## plot
# create df
pred_summ, pred_quant = describe(pred, q=[0.025, 0.05, 0.5, 0.95, 0.975])
pred_summ_prior, pred_quant_prior = describe(pred_prior, q=[0.025, 0.05, 0.5, 0.95, 0.975])

# get DFs
## posterior
df_pred_summ = DataFrame(pred_summ)
df_pred_quant = DataFrame(pred_quant)
df_pred_summquant = @transform(hcat(df_pred_summ, select(df_pred_quant, Not(:parameters))), :distribution = "posterior")

## prior
df_pred_summ_prior = DataFrame(pred_summ_prior)
df_pred_quant_prior = DataFrame(pred_quant_prior)
df_pred_summquant_prior = @transform(hcat(df_pred_summ_prior, select(df_pred_quant_prior, Not(:parameters))), :distribution = "prior")

## join
df_pred_summquant_all = vcat(df_pred_summquant, df_pred_summquant_prior)

# save
CSV.write(joinpath(resDir, "df_pred_summquant_all.csv"), df_pred_summquant_all)

####

df1 = hcat(select(DataFrame(pred_summ), Not(:parameters)), DataFrame(pred_quant))
#df2 = hcat(@select(dat_train[dat_train.BLQ .== 0,:], :ID,:TIME,:AMT,:DV,:DOSE,:BLQ), df1)
df2 = @select(dat_train, :ID, :TIME, :AMT, :DV, :DOSE, :BLQ)
bins = [0, 1, 1.5, 2, 3, 4, 5, 6, 8, 12, 20, 24, 48, 72, 96]
labels = string.(1:length(bins))
@transform!(df2, :bins = cut(:TIME, bins, labels=labels, extend=true))

#df3 = hcat(@rsubset(df_pred_summquant_all, :distribution=="posterior"), df2[df2.BLQ.==0, :])  # uncomment when reading CSV file
df3 = hcat(df1, df2[df2.BLQ.==0, :])

# plot
## facetted
pl = AlgebraOfGraphics.data(df3)
l1 = mapping(:TIME => "Time (h)", :DV => "Concentration (ng/mL)", layout=:ID => nonnumeric)
l2 = mapping(:TIME => "Time (h)", "50.0%" => "Concentration (ng/mL)", layout=:ID => nonnumeric) * visual(Lines, color=:red)
l4 = mapping(:TIME => "Time (h)", "2.5%" => "Concentration (ng/mL)", "97.5%" => "Concentration (ng/mL)", layout=:ID => nonnumeric) * visual(Band; alpha=0.5, color=:lightblue)
pl_PPCbyID = draw(pl * (l1 + l2 + l4), facet=(; linkyaxes=:none))
pl_PPCbyID_log = draw(pl * (l1 + l2 + l4), facet=(; linkyaxes=:none), axis=(yscale=log10,))

## summary
df4 = @chain df2 begin
    @rsubset(:BLQ == 0)
    @rtransform(:DNDV = :DV ./ :DOSE)

    groupby(:bins)
    @transform(:loDV = quantile(:DNDV, 0.05),
        :medDV = quantile(:DNDV, 0.5),
        :hiDV = quantile(:DNDV, 0.95))
end
df_pred = DataFrame(pred)

df_ppc_pred = @chain df_pred begin
    DataFramesMeta.stack(3:ncol(df_pred))
    @orderby(:iteration, :chain)
    hcat(select(repeat(df3, nsampl * nchains), [:ID, :TIME, :DOSE, :BLQ]))
    @transform(:DNPRED = :value ./ :DOSE,
        :bins = cut(:TIME, bins, labels=labels, extend=true))

    groupby([:iteration, :chain, :bins])
    @transform(:lo = quantile(:DNPRED, 0.05),
        :med = quantile(:DNPRED, 0.5),
        :hi = quantile(:DNPRED, 0.95))

    groupby(:bins)
    @transform(:loLo = quantile(:lo, 0.025),
        :medLo = quantile(:lo, 0.5),
        :hiLo = quantile(:lo, 0.975),
        :loMed = quantile(:med, 0.025),
        :medMed = quantile(:med, 0.5),
        :hiMed = quantile(:med, 0.975),
        :loHi = quantile(:hi, 0.025),
        :medHi = quantile(:hi, 0.5),
        :hiHi = quantile(:hi, 0.975))
end
df_ppc_pred2 = combine(first, groupby(df_ppc_pred, [:bins, :DOSE]))

# save
CSV.write(joinpath(resDir, "df4.csv"), df4)
CSV.write(joinpath(resDir, "df_ppc_pred2.csv"), df_ppc_pred2)

l1 = AlgebraOfGraphics.data(df4) * mapping(:TIME => "Time (h)", :DNDV => "Dose-normalized concentration (ng/mL/mg)") * visual(Scatter);
l2 = AlgebraOfGraphics.data(df4) * mapping(:TIME => "Time (h)", :medDV => "Dose-normalized concentration (ng/mL/mg)"; group=:ID => nonnumeric) * visual(Lines, linestyle=:dash);
l3 = AlgebraOfGraphics.data(df4) * mapping(:TIME => "Time (h)", :loDV => "Dose-normalized concentration (ng/mL/mg)"; group=:ID => nonnumeric) * visual(Lines, linestyle=:dash);
l4 = AlgebraOfGraphics.data(df4) * mapping(:TIME => "Time (h)", :hiDV => "Dose-normalized concentration (ng/mL/mg)"; group=:ID => nonnumeric) * visual(Lines, linestyle=:dash);
l5 = AlgebraOfGraphics.data(df_ppc_pred2) * mapping(:TIME => "Time (h)", :loLo => "Dose-normalized concentration (ng/mL/mg)", :hiLo => "Dose-normalized concentration (ng/mL/mg)") * visual(Band; alpha=0.5, color=:lightblue);
l6 = AlgebraOfGraphics.data(df_ppc_pred2) * mapping(:TIME => "Time (h)", :loMed => "Dose-normalized concentration (ng/mL/mg)", :hiMed => "Dose-normalized concentration (ng/mL/mg)") * visual(Band; alpha=0.5, color=:blue);
l7 = AlgebraOfGraphics.data(df_ppc_pred2) * mapping(:TIME => "Time (h)", :loHi => "Dose-normalized concentration (ng/mL/mg)", :hiHi => "Dose-normalized concentration (ng/mL/mg)") * visual(Band; alpha=0.5, color=:lightblue);
l8 = AlgebraOfGraphics.data(df_ppc_pred2) * mapping(:TIME => "Time (h)", :medMed => "Dose-normalized concentration (ng/mL/mg)") * visual(Lines, color=:blue);

pl_summ = draw(l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)
pl_summ_log = draw(l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8, axis=(yscale=log10,))


###########################
###########################

###########

## external validation ##

dat_test = @rsubset(dat2, !(:ID in ids_train))
ids_test = sort(unique(dat_test.ID))
dat_test.DV[dat_test.DV.<=0.0] .= 1e-12  # avoid 0
nSubj_test = length(unique(dat_test.ID))
blq_test = dat_test.BLQ
L = 10.0  # assuming loq is 10

# sample etas
#df_params = DataFrame(ch)[:,3:10]
ηs_ka = reshape(rand(Normal(0.0, 1.0), nSubj_test * nsampl), nsampl, nSubj_test)
ηs_CL = reshape(rand(Normal(0.0, 1.0), nSubj_test * nsampl), nsampl, nSubj_test)
ηs_V2 = reshape(rand(Normal(0.0, 1.0), nSubj_test * nsampl), nsampl, nSubj_test)

# get pk pars
pars_test = Array{Float64}(undef, nsampl, length(p), nSubj_test)
for i in 1:nsampl
    for j in 1:nSubj_test
        NN_ = re(w[i, :, 1])
        pars_test[i, :, j] = NN_(covsn[ids_test, :][j, :])
        # add random effects
        pars_test[i, 1, j] = pars_test[i, 1, j] * exp.(ω_ka[i, 1, 1] .* ηs_ka[i, j])
        pars_test[i, 2, j] = pars_test[i, 2, j] * exp.(ω_CL[i, 1, 1] .* ηs_CL[i, j])
        pars_test[i, 3, j] = pars_test[i, 3, j] * exp.(ω_V2[i, 1, 1] .* ηs_V2[i, j])
        #push!(pars, NN_(covsn[j,:]))
    end
end

# simulate
pred_test = Array{Float64}(undef, nrow(dat_test), nsampl)

for i in 1:nsampl
    preds_ = []
    for j in 1:nSubj_test
        ps_ = pars_test[i, :, j]
        u0_ = [doses[ids_test][j], 0.0, 0.0]
        times_ = times[ids_test][j]
        tspan_ = (0.0, times_[end])
        prob_ = remake(prob, p=ps_, u0=u0_, tspan=tspan_, saveat=times_)
        sol_ = solve(prob_, Tsit5())[2,:]
        append!(preds_, sol_)
    end
    pred_test[:, i] = preds_
end

df_test1 = DataFrame(pred_test, :auto)
df_test2 = DataFramesMeta.stack(df_test1)

# get Stats
bins = [0, 1, 1.5, 2, 3, 4, 5, 6, 8, 12, 20, 24, 48, 72, 96]

## obs
@transform!(dat_test, :bins = cut(:TIME, bins, labels=labels, extend=true))
dat_test2 = @chain dat_test begin
    @rtransform(:DNDV = :DV ./ :DOSE)

    @rsubset(:BLQ == 0)

    groupby(:bins)
    @transform(:loDV = quantile(:DNDV, 0.05),
        :medDV = quantile(:DNDV, 0.5),
        :hiDV = quantile(:DNDV, 0.95))
end

dat_test3 = dat_test2[dat_test2.BLQ.==0, :]

## pred
## get data
df_ppc_pred_test = @chain df_test2 begin
    hcat(select(repeat(dat_test, nsampl), [:ID, :TIME, :DOSE, :BLQ]))
    @transform(:DNPRED = :value ./ :DOSE,
        :bins = cut(:TIME, bins, labels=labels, extend=true))

    @rsubset(:BLQ == 0)

    groupby([:variable, :bins])
    @transform(:lo = quantile(:DNPRED, 0.05),
        :med = quantile(:DNPRED, 0.5),
        :hi = quantile(:DNPRED, 0.95))

    groupby(:bins)
    @transform(:loLo = quantile(:lo, 0.025),
        :medLo = quantile(:lo, 0.5),
        :hiLo = quantile(:lo, 0.975),
        :loMed = quantile(:med, 0.025),
        :medMed = quantile(:med, 0.5),
        :hiMed = quantile(:med, 0.975),
        :loHi = quantile(:hi, 0.025),
        :medHi = quantile(:hi, 0.5),
        :hiHi = quantile(:hi, 0.975))
end

df_ppc_pred_test2 = combine(first, groupby(df_ppc_pred_test, [:bins, :DOSE]))

df_ppc_pred_test_facet = @chain df_test2 begin
    hcat(select(repeat(dat_test, nsampl), [:ID, :TIME, :DOSE, :BLQ]))
    @transform(:bins = cut(:TIME, bins, labels=labels, extend=true))

    @rsubset(:BLQ == 0)

    groupby([:ID, :bins])
    @transform(:lo = quantile(:value, 0.05),
        :med = quantile(:value, 0.5),
        :hi = quantile(:value, 0.95),
        :avg = mean(:value))
end
df_ppc_pred_test_facet2 = hcat(@rsubset(df_ppc_pred_test_facet, :variable == "x1"), DataFrame(DV=dat_test.DV[dat_test.BLQ.==0]))

CSV.write(joinpath(resDir, "df_ppc_pred_test2.csv"), df_ppc_pred_test2)
CSV.write(joinpath(resDir, "df_ppc_pred_test_facet2.csv"), df_ppc_pred_test_facet2)

# plot
## facetted by ID
## facetted
pl = AlgebraOfGraphics.data(df_ppc_pred_test_facet2)
l1 = mapping(:TIME => "Time (h)", :DV => "Concentration (ng/mL)", layout=:ID => nonnumeric)
l2 = mapping(:TIME => "Time (h)", :med => "Concentration (ng/mL)", layout=:ID => nonnumeric) * visual(Lines, color=:red)
l4 = mapping(:TIME => "Time (h)", :lo => "Concentration (ng/mL)", :hi => "Concentration (ng/mL)", layout=:ID => nonnumeric) * visual(Band; alpha=0.5, color=:lightblue)
pl_PPCbyID_test = draw(pl * (l1 + l2 + l4), facet=(; linkyaxes=:none))
pl_PPCbyID_test_log = draw(pl * (l1 + l2 + l4), axis=(yscale=log10,), facet=(; linkyaxes=:none))

## summary
l1 = AlgebraOfGraphics.data(dat_test3) * mapping(:TIME => "Time (h)", :DNDV => "Dose-normalized concentration (ng/mL/mg)") * visual(Scatter);
l2 = AlgebraOfGraphics.data(dat_test3) * mapping(:TIME => "Time (h)", :medDV => "Dose-normalized concentration (ng/mL/mg)"; group=:ID => nonnumeric) * visual(Lines, linestyle=:dash);
l3 = AlgebraOfGraphics.data(dat_test3) * mapping(:TIME => "Time (h)", :loDV => "Dose-normalized concentration (ng/mL/mg)"; group=:ID => nonnumeric) * visual(Lines, linestyle=:dash);
l4 = AlgebraOfGraphics.data(dat_test3) * mapping(:TIME => "Time (h)", :hiDV => "Dose-normalized concentration (ng/mL/mg)"; group=:ID => nonnumeric) * visual(Lines, linestyle=:dash);
l5 = AlgebraOfGraphics.data(df_ppc_pred_test2) * mapping(:TIME => "Time (h)", :loLo => "Dose-normalized concentration (ng/mL/mg)", :hiLo => "Dose-normalized concentration (ng/mL/mg)") * visual(Band; alpha=0.5, color=:lightblue);
l6 = AlgebraOfGraphics.data(df_ppc_pred_test2) * mapping(:TIME => "Time (h)", :loMed => "Dose-normalized concentration (ng/mL/mg)", :hiMed => "Dose-normalized concentration (ng/mL/mg)") * visual(Band; alpha=0.5, color=:blue);
l7 = AlgebraOfGraphics.data(df_ppc_pred_test2) * mapping(:TIME => "Time (h)", :loHi => "Dose-normalized concentration (ng/mL/mg)", :hiHi => "Dose-normalized concentration (ng/mL/mg)") * visual(Band; alpha=0.5, color=:lightblue);
l8 = AlgebraOfGraphics.data(df_ppc_pred_test2) * mapping(:TIME => "Time (h)", :medMed => "Dose-normalized concentration (ng/mL/mg)") * visual(Lines, color=:blue);

pl_summ_test = draw(l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)
pl_summ_log_test = draw(l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8, axis=(yscale=log10,))

###############
###############

## Shaply ##

NN_trained = re(w[1, :, 1])
X = DataFrame(covsn, names(contcovs))
#y = predict_cl(NN_trained, X)

ws = vcat(w[:, :, 1], w[:, :, 2])

#------------------------------------------------------------------------------
# ShapML setup.
explain = deepcopy(X)  #X[ids_train,:]  #copy(boston[1:300, :]) # Compute Shapley feature-level predictions for 300 instances.

reference = deepcopy(X)  #copy(boston)  # An optional reference population to compute the baseline prediction.

sample_size = 100  # Number of Monte Carlo samples.

#=
## serial ##

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
function predict_cl(model, data)
    data2 = Array(data)
    [model(x)[2] for x in eachrow(data2)]
end

# Create a wrapper function that takes the following positional arguments: (1) a
# trained ML model from any Julia package, (2) a DataFrame of model features. The
# function should return a 1-column DataFrame of predictions--column names do not matter.
function predict_function(model, data)
    data_pred = DataFrame(y_pred = predict_cl(model, data))
    return data_pred
end

data_shaps = []
@time for i in 1:nrow(ws)
    NN_trained = re(Array(ws[i,:]))

    #------------------------------------------------------------------------------
    # Compute stochastic Shapley values.
    @time data_shap = ShapML.shap(explain = explain,
    reference = reference,
    model = NN_trained,
    predict_function = predict_function,
    sample_size = sample_size,
    seed = 1
    )

    push!(data_shaps, data_shap)
end
=#

#-----------------#

## parallel ##

#=
## install packages on all workers
@everywhere begin 
    using Pkg
    Pkg.add(["MLJ", "ShapML","DataFramesMeta", "Flux"])
end
=#

@everywhere begin
    using ShapML
    using DataFramesMeta
    using Flux
end

# for CL
@everywhere function predict_cl(model, data)
    data2 = Array(data)
    [model(x)[2] for x in eachrow(data2)]
end

@everywhere function predict_function(model, data)
    data_pred = DataFrame(y_pred=predict_cl(model, data))
    return data_pred
end

data_shaps = []
@time for i in 1:length(ws[:, 1])
    NN_trained = re(Array(ws[i, :]))

    data_shap = ShapML.shap(explain=explain,
        reference=reference,
        model=NN_trained,
        predict_function=predict_function,
        sample_size=sample_size,
        parallel=:samples,
        seed=1
    )

    @transform!(data_shap, :sample = i)

    push!(data_shaps, data_shap)
end

# for V2
@everywhere function predict_v2(model, data)
    data2 = Array(data)
    [model(x)[3] for x in eachrow(data2)]
end

@everywhere function predict_function_v2(model, data)
    data_pred = DataFrame(y_pred=predict_v2(model, data))
    return data_pred
end

data_shaps_v2 = []
@time for i in 1:length(ws[:, 1])
    NN_trained = re(Array(ws[i, :]))

    data_shap_v2 = ShapML.shap(explain=explain,
        reference=reference,
        model=NN_trained,
        predict_function=predict_function_v2,
        sample_size=sample_size,
        parallel=:samples,
        seed=1
    )

    @transform!(data_shap_v2, :sample = i)

    push!(data_shaps_v2, data_shap_v2)
end

##---------------------##                        

# post-processing

## CL

data_shaps2 = vcat(data_shaps...)

data_shaps3 = @chain data_shaps2 begin
    groupby([:feature_name, :index])
    @transform(:lo = quantile(:shap_effect, 0.025),
        :med = quantile(:shap_effect, 0.5),
        :hi = quantile(:shap_effect, 0.975))
    groupby([:feature_name, :sample])
    @transform(:mae = mean(abs.(:shap_effect)))
end

data_shaps4 = combine(first, groupby(data_shaps3, [:feature_name, :index]))
data_shaps4_mae = combine(first, groupby(data_shaps3, [:feature_name, :sample]))

data_shaps5 = sort(data_shaps4, [:feature_name, :feature_value])

data_shaps5_mae = @chain data_shaps4_mae begin
    groupby(:feature_name)
    @transform(:loMAE = quantile(:mae, 0.05),
        :medMAE = quantile(:mae, 0.5),
        :hiMAE = quantile(:mae, 0.95))
end

data_plot_mae = sort(combine(first, groupby(data_shaps5_mae, :feature_name)), :medMAE)

CSV.write(joinpath(resDir, "df_shap_mae.csv"), data_plot_mae)
CSV.write(joinpath(resDir, "df_shaps4_mae.csv"), data_shaps4_mae)
CSV.write(joinpath(resDir, "df_shaps5.csv"), data_shaps5)


# plot
## mean shap values
pl = AlgebraOfGraphics.data(data_plot_mae)
l1 = mapping(:feature_name => :Covariate, :medMAE => "SHAP effect on CL") * visual(BarPlot)
pl_shap_bar = draw(pl * l1)

## violin
pl = AlgebraOfGraphics.data(data_shaps4_mae)
l1 = mapping(:feature_name => :Covariate, :mae => "SHAP effect on CL") * visual(Violin)
pl_shap_violin = draw(pl * l1)

# how changing feature affect prediction
pl = AlgebraOfGraphics.data(data_shaps5)
l1 = mapping(:feature_value => "Covariate value", :med => "SHAP effect on CL", layout=:feature_name) * (smooth() + visual(Scatter))
l2 = mapping(:feature_value => "Covariate value", :lo => "SHAP effect on CL", :hi, layout=:feature_name) * visual(Band, alpha=0.5, color=:lightblue)
pl_shap_change = draw(pl * (l1 + l2))


## covariate interaction plot

data_shaps5_interaction_alb = select(@rsubset(data_shaps5, :feature_name == "ALB"), :index, :med => :ALB)
data_shaps5_interaction = @chain data_shaps5 begin 
    @rsubset(:feature_name == "WT")
    leftjoin(data_shaps5_interaction_alb, on = :index)
end

pl = AlgebraOfGraphics.data(data_shaps5_interaction)
l1 = mapping(:feature_value => "Covariate value", :med => "SHAP effect on CL", layout=:feature_name, color = :ALB) * (smooth() + visual(Scatter))
#l2 = mapping(:feature_value => "Covariate value", :lo => "SHAP effect on CL", :hi, layout=:feature_name) * visual(Band, alpha=0.5, color=:lightblue)
pl_shap_change_interaction = draw(pl * (l1))


###

## V2

data_shaps2_v2 = vcat(data_shaps_v2...)

data_shaps3_v2 = @chain data_shaps2_v2 begin
    groupby([:feature_name, :index])
    @transform(:lo = quantile(:shap_effect, 0.025),
        :med = quantile(:shap_effect, 0.5),
        :hi = quantile(:shap_effect, 0.975))
    groupby([:feature_name, :sample])
    @transform(:mae = mean(abs.(:shap_effect)))
end

data_shaps4_v2 = combine(first, groupby(data_shaps3_v2, [:feature_name, :index]))
data_shaps4_mae_v2 = combine(first, groupby(data_shaps3_v2, [:feature_name, :sample]))

data_shaps5_v2 = sort(data_shaps4_v2, [:feature_name, :feature_value])

data_shaps5_mae_v2 = @chain data_shaps4_mae_v2 begin
    groupby(:feature_name)
    @transform(:loMAE = quantile(:mae, 0.05),
        :medMAE = quantile(:mae, 0.5),
        :hiMAE = quantile(:mae, 0.95))
end

data_plot_mae_v2 = sort(combine(first, groupby(data_shaps5_mae_v2, :feature_name)), :medMAE)

# save
CSV.write(joinpath(resDir, "df_shap_mae_v2.csv"), data_plot_mae_v2)
CSV.write(joinpath(resDir, "df_shaps4_v2.csv"), data_shaps4_mae_v2)
CSV.write(joinpath(resDir, "df_shaps5_v2.csv"), data_shaps5_v2)

# plot
## mean shap values
pl = AlgebraOfGraphics.data(data_plot_mae_v2)
l1 = mapping(:feature_name => :Covariate, :medMAE => "SHAP effect on V2") * visual(BarPlot)
pl_shap_bar_v2 = draw(pl * l1)

## violin
pl = AlgebraOfGraphics.data(data_shaps4_mae_v2)
l1 = mapping(:feature_name => :Covariate, :mae => "SHAP effect on V2") * visual(Violin)
pl_shap_violin_v2 = draw(pl * l1)

# how changing feature affect prediction
pl = AlgebraOfGraphics.data(data_shaps5_v2)
l1 = mapping(:feature_value => "Covariate value", :med => "SHAP effect on V2", layout=:feature_name) * (smooth() + visual(Scatter))
l2 = mapping(:feature_value => "Covariate value", :lo => "SHAP effect on V2", :hi, layout=:feature_name) * visual(Band, alpha=0.5, color=:lightblue)
pl_shap_change_v2 = draw(pl * (l1 + l2))

####################
####################

## save results
### figures
save(joinpath(resDir, "data_profiles.pdf"), pl_data_log)
save(joinpath(resDir, "mcmcchains.pdf"), pl_chains_sample)
save(joinpath(resDir, "mcmcchains_prior.pdf"), pl_chains_sample_prior)
save(joinpath(resDir, "mcmcdensity.pdf"), pl_density_sample)
save(joinpath(resDir, "mcmcdensity_prior.pdf"), pl_density_sample_prior)
save(joinpath(resDir, "PPC_byID.pdf"), pl_PPCbyID)
save(joinpath(resDir, "PPC_byID_external.pdf"), pl_PPCbyID_test)
save(joinpath(resDir, "PPC_byID_log.pdf"), pl_PPCbyID_log)
save(joinpath(resDir, "PPC_byID_external_log.pdf"), pl_PPCbyID_test_log)
save(joinpath(resDir, "PPC_summ.pdf"), pl_summ)
save(joinpath(resDir, "PPC_summ_log.pdf"), pl_summ_log)
save(joinpath(resDir, "PPC_summ_external.pdf"), pl_summ_test)
save(joinpath(resDir, "PPC_summ_log_external.pdf"), pl_summ_log_test)
save(joinpath(resDir, "shap_bar.pdf"), pl_shap_bar)
save(joinpath(resDir, "shap_violin.pdf"), pl_shap_violin)
save(joinpath(resDir, "shap_change.pdf"), pl_shap_change)
save(joinpath(resDir, "shap_bar_v2.pdf"), pl_shap_bar_v2)
save(joinpath(resDir, "shap_violin_v2.pdf"), pl_shap_violin_v2)
save(joinpath(resDir, "shap_change_v2.pdf"), pl_shap_change_v2)
save(joinpath(resDir, "shap_change_interaction.pdf"), pl_shap_change_interaction)


## everything
JLD2.save(joinpath(resDir, "results.jld2"),
    "chains", ch, "chains_prior", ch_prior, "p_map_array", p_map_array, #"p_mle_array",p_mle_array, 
    "pars_ind", pars_ind, "pars_pop", pars_pop, "data", dat2, "nsample", nsampl, "adapt_delta", adapt_delta, "nchains", nchains,
    "pred", pred, "pred_prior", pred_prior, "nSubj_train", nSubj_train, "nSubj_test", nSubj_test, "NN", NN, "covs", covs, "covsn", covsn)