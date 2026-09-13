import pytest
import torch
from notebooks.bnn_from_scratch import BayesianLinear, BayesianNeuralNetwork, negative_elbo, predictive_moments, make_data, fit


def test_analytic_kl_matches_torch_distributions():
    layer = BayesianLinear(2, 3, prior_sigma=0.7).double()
    from torch.distributions import Normal, kl_divergence
    expected = sum(kl_divergence(Normal(mu, layer.scale(rho)), Normal(torch.zeros_like(mu), layer.prior_sigma)).sum()
                   for mu, rho in ((layer.weight_mu, layer.weight_rho), (layer.bias_mu, layer.bias_rho)))
    torch.testing.assert_close(layer.kl_divergence(), expected)


def test_elbo_averages_log_likelihood_not_predictions():
    draws = torch.tensor([[[-1.]], [[1.]]])
    y = torch.zeros(1, 1)
    actual = negative_elbo(draws, y, torch.tensor(4.), dataset_size=2, noise_std=1.)
    expected = 0.5 * (1 + torch.log(torch.tensor(2 * torch.pi))) + 2
    torch.testing.assert_close(actual, expected)
    assert actual > negative_elbo(draws.mean(0, keepdim=True), y, torch.tensor(4.), 2, 1.)


def test_minibatch_scaling():
    draws = torch.tensor([[[1.], [2.], [3.], [4.]], [[2.], [1.], [4.], [3.]]])
    y = torch.zeros(4, 1)
    kl = torch.tensor(3.)
    full = negative_elbo(draws, y, kl, 4)
    batches = [negative_elbo(draws[:, s:s+2], y[s:s+2], kl, 4) for s in (0, 2)]
    torch.testing.assert_close(full, sum(batches) / 2)


def test_gradient_and_device_dtype():
    model = BayesianNeuralNetwork().double()
    x, y, _, _ = make_data(n_train=5)
    draws = model(x.double(), 3)
    assert draws.dtype == torch.float64
    assert draws.device == x.device
    loss = negative_elbo(draws, y.double(), model.kl_divergence(), 5)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    mean, epistemic, total = predictive_moments(model, x.double(), 30, 0.3)
    torch.testing.assert_close(total - epistemic, torch.full_like(mean, 0.09))
    assert torch.all(epistemic >= 0)


def test_small_training_improves_unseen_prediction():
    torch.manual_seed(7)
    x, y, xt, yt = make_data(seed=7, n_train=80)
    model = BayesianNeuralNetwork(hidden_dim=10)
    before = ((predictive_moments(model, xt, 100)[0] - yt)**2).mean()
    fit(model, x, y, steps=300)
    after = ((predictive_moments(model, xt, 100)[0] - yt)**2).mean()
    assert after < before


def test_invalid_arguments():
    with pytest.raises(ValueError):
        BayesianLinear(1, 1, prior_sigma=0)
    model = BayesianNeuralNetwork()
    with pytest.raises(ValueError):
        model(torch.ones(1,1), 0)
    with pytest.raises(ValueError):
        negative_elbo(torch.ones(1,2,1), torch.zeros(2,1), torch.tensor(0.), 1)
    with pytest.raises(ValueError):
        negative_elbo(torch.ones(1,1,1), torch.zeros(1,1), torch.tensor(0.), 1, noise_std=0)
    with pytest.raises(ValueError):
        negative_elbo(torch.ones(1,1), torch.zeros(1,1), torch.tensor(0.), 1)
    with pytest.raises(ValueError):
        predictive_moments(model, torch.ones(1,1), num_samples=1)
