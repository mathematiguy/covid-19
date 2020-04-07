data {
  // The number of data points
  int<lower=1> N;
  // The observations
  int y[N];
  // The size of the population
  int P;
  // The number of timesteps to forecast
  int<lower=0> T;

}
parameters {
  // The local level
  vector[N] mu;
  // Level noise
  vector<lower=0>[N] sigma_mu;
}
model {
  // Priors
  sigma_mu[1] ~ exponential(inv_logit(1. / logit(1. / P)));
  mu[1] ~ normal(inv_logit(1. / logit(1. / P)), sigma_mu[1]);

  // Update the state space
  for (t in 2:N) {
    sigma_mu[t] ~ exponential(inv_logit(1. / mu[t-1]));
    mu[t] ~ normal(inv_logit(1. / (mu[t-1])), sigma_mu[t]);
  }

  for (t in 1:N) {
    y[t] ~ binomial_logit(P, mu[t]);
  }

}
generated quantities {
  // The posterior predictions for y
  vector[N] y_pred;

  for (t in 1:N) {
    y_pred[t] = binomial_rng(P, inv_logit(mu[t]));
  }

}
