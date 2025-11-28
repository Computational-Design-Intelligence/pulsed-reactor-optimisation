from utils import *
#from jax.config import config
from jax import config
from jax import numpy as jnp
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

def compute_mutual_info(gp, x, x_cond):
    # Compute the covariance matrix of the GP at x and the training point
    # kern = gpx.kernels.Matern52()
    # kern.init_params(gp["learned_params"]["kernel"])
    # K = kern(gp["learned_params"]["kernel"], x,x_cond)
    # info = jnp.sqrt(1-K**2)
    # if correlation between data is LOW then info is HIGH
    kern= gpx.kernels.Matern52()
    kern.init_params(gp["learned_params"]["kernel"])
    k= kern(gp["learned_params"]["kernel"], x, x_cond)  # Cross-covariance between x and x_cond
    info= jnp.sqrt(1 - k**2)
    return info


def exp_design_function(x, gp, cost_gp, fid_high, gamma, cost_offset):
    #obtain predicted cost 
    c_m,c_v = inference(cost_gp, jnp.array([x]))

    x_cond = jnp.copy(x)
    for i in range(len(fid_high)):
        i += 1
        x_cond = x_cond.at[-i].set(fid_high[-i])

    f_m, f_v = inference(gp, jnp.array([x_cond]))
    f_std= jnp.sqrt(jnp.maximum(f_v, 0.0))
    c_m= c_m[0]+cost_offset
    # f_v = jnp.sqrt(f_v[0,0])
    # c_m = c_m[0]+cost_offset


    # # Compute mutual information between high fidelity model and new observation
    mutual_info = -compute_mutual_info(gp, x,x_cond)

    val = ((-f_v)/(c_m*mutual_info))[0]
    return val


def exp_design_hf(x, gp):
    #obtain predicted cost 

    f_m, f_v = inference(gp, jnp.array([x]))
    f_v = jnp.sqrt(f_v[0,0])
    val = ((-f_v))
    return val


def aquisition_function(x, gp, cost_gp, fid_high, gamma, beta, cost_offset):
    # obtain predicted cost 
    cost, cost_var = inference(cost_gp, jnp.array([x]))
    # fixing fidelity
    x= jnp.array(x)
    for i in range(len(fid_high)):
        i += 1
        x = x.at[-i].set(fid_high[-i])
    # obtain predicted objective
    mean, cov = inference(gp, jnp.array([x]))
    # weighted acquisition function. note cost offset required for non -inf values
    result= (-((mean[0] + beta * cov[0]) / ((gamma * (cost[0] - cost_offset)))))
    return jnp.squeeze(result)


def greedy_function(x, gp, fid_high):
    # fixing fidelity
    for i in range(len(fid_high)):
        x = jnp.append(x, fid_high[i])
    # predict objective
    mean, cov = inference(gp, jnp.array([x]))
    return -mean[0]


def train_gp(inputs, outputs, ms):
    # creating a set of initial GP hyper parameters (log-spaced)
    init_params = lhs(
        np.array([[0.1, 10] for i in range(len(inputs[0, :]))]), ms, log=True
    )
    # defining dataset
    D = gpx.Dataset(X=inputs, y=outputs)
    best_loss = np.inf
    best_posterior = None
    best_history = None
    best_likelihood = None
    # for each intital list of hyperparameters
    best_nll = 1e30
    base_key = jax.random.PRNGKey(0)
    for p in init_params:
        
        subkey, base_key = jax.random.split(base_key)
        # define kernel function 
        kern = gpx.kernels.Matern52(active_dims=[i for i in range(D.in_dim)])
        # define prior GP
        meanf = gpx.mean_functions.Zero()
        prior = gpx.gps.Prior(mean_function=meanf, kernel=kern)
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)
        # Bayes rule
        posterior = prior * likelihood
        objective= gpx.objectives.ConjugateMLL(negative=True)
        opt= ox.adam(learning_rate=0.05)
        # # negative log likelihood
        # mll = jit(posterior.marginal_log_likelihood(D, negative=True))
        # # initialise optimizer
        # opt = ox.adam(learning_rate=0.05)
        
        # define intial hyper parameters 
        #parameter_state = gpx.initialise(posterior)
        # parameter_state.trainables['likelihood']['obs_noise'] = False
        # parameter_state.params['likelihood']['obs_noise'] = 0
        # parameter_state.params["kernel"]["lengthscale"] = p

        # run optimiser
        # inference_state = gpx.fit(mll, parameter_state, opt, num_iters=50000,log_rate=100)
        #inference_state = gpx.fit(mll, parameter_state, opt, num_iters=25000,log_rate=100)
        # get last NLL value
        # # get last NLL value that isn't a NaN
        # inf_history = inference_state.history
        # # remore nan values
        # inf_history = [x for x in inf_history if str(x) != "nan"]
        # nll = float(inf_history[-1])
        # # if this is the best, then store this 
        # if nll < best_nll:
        #     best_nll = nll
        #     best_inference_state = inference_state
        #     best_likelihood = likelihood
        #     best_posterior = posterior
        posterior_opt, history = gpx.fit(
            model=posterior,
            objective=objective,
            train_data=D,
            optim=opt,
            num_iters=250,
            log_rate=100,
            key=subkey,
        )

        # Final loss from history (filter NaNs like before)
        clean_hist = [float(v) for v in history if str(v) != "nan"]
        if not clean_hist:
            continue
        final_loss = clean_hist[-1]

        if final_loss < best_loss:
            best_loss = final_loss
            best_posterior = posterior_opt
            best_history = clean_hist

    #learned_params, _ = best_inference_state.unpack()
    learned_params= getattr(best_posterior, "params", None)
    # return everything that defines a GP!
    return best_posterior, learned_params, D, likelihood




def inference(gp, inputs):
    # perform inference on a GP given inputs

    # obtain features from GP dict
    # posterior = gp["posterior"]
    # learned_params = gp["learned_params"]
    # D = gp["D"]
    # likelihood = gp["likelihood"]

    # # definte distributions
    # latent_distribution = posterior(learned_params, D)(inputs)
    # predictive_distribution = likelihood(learned_params, latent_distribution)
    # predictive_mean = predictive_distribution.mean()
    # predictive_cov = predictive_distribution.covariance()
    # return predictive_mean, predictive_cov
    model= gp["model"]
    D = gp["data"]
    Xnew= jnp.atleast_2d(inputs)
    pred= model.predict(Xnew, train_data=D)
    mean= jnp.atleast_1d(jnp.squeeze(pred.mean()))
    var= jnp.atleast_1d(jnp.squeeze(pred.variance()))
    return mean, var

def build_gp_dict(posterior, learned_params, D, likelihood):
    # build a dictionary to store features to make everything cleaner
    # gp_dict = {}
    # gp_dict["posterior"] = posterior
    # gp_dict["learned_params"] = learned_params
    # gp_dict["D"] = D
    # gp_dict["likelihood"] = likelihood
    return {"model": posterior, "data": D, "likelihood": likelihood}
