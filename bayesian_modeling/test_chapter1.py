from scipy import stats

def test_prior_beta_11():
    θ = 0.5
    prior = stats.beta(1, 1).pdf(θ).round(2)
    assert prior == 1

def test_likelihood():
    θ = 0.5
    Y = stats.bernoulli(0.7).rvs(20)
    like = stats.bernoulli(θ).pmf(Y).prod()
    assert like.round(2) == 0.0
