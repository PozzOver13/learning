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


def test_metropolis_hastings_algo():
    θ_target = 0.7
    θ = 0.5
    Y = stats.bernoulli(θ_target).rvs(5)
    prior = stats.beta(1, 1).pdf(θ)
    like = stats.bernoulli(θ).pmf(Y).prod()
    p2 = prior * like

    # Proposal distribution
    # θ_mh = 0.01
    θ_mh = 0.63
    # θ_mh = stats.norm(θ, 0.05).rvs(1)

    prior = stats.beta(1, 1).pdf(θ_mh)
    like = stats.bernoulli(θ_mh).pmf(Y).prod()
    p1 = prior * like

    pa = p1 / p2

    condition = pa > 0.5  # stats.uniform(0, 1).rvs(1)

    print("\n----\nMetropolis-Hastings Algorithm")
    print(f"initial θ: {θ}, sampled θ: {θ_mh}")
    print(f"initial likelihood: {p2.round(2)}, sampled likelihood: {p1.round(2)}")
    print(f"likelihood ratio: {pa.round(2)}")
    print("accepted = ", condition)

    assert condition == True
