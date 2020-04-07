data {
  // The number of data points
  int<lower=1> N;
  // The observations
  int y[N];
  // The size of the population
  int P;
}

generated quantities {
  // The local level
  vector[N] mu;
  // The local variance
  vector<lower=0>[N] mu_sigma;
  // Level noise
  vector[N] y_prior;

  // Priors
  mu_sigma[1] = exponential_rng(1 / inv_logit(1. / logit(1. / P)));
  mu[1] = normal_rng(logit(1. / P), mu_sigma[1]);

  // Update the state space
  for (t in 2:N) {
    mu_sigma[t] = exponential_rng(1 / inv_logit(1. / mu[t-1]));
    mu[t] = normal_rng(mu[t-1], mu_sigma[t]);
  }

  // Update the output
  for (t in 1:N) {
    y_prior[t] = binomial_rng(P, inv_logit(mu[t]));
  }

}
