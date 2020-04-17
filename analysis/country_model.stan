data {
  // The number of data points
  int<lower=1> N;
  // The observations
  int y[N];
  // The size of the population
  int P;
  // The number of timesteps to forecast
  int<lower=1> T;

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
    mu[t] ~ normal(inv_logit(1. / mu[t-1]), sigma_mu[t]);
  }

  for (t in 1:N) {
    y[t] ~ binomial_logit(P, mu[t]);
  }

}
generated quantities {
  // The posterior predictions for y
  vector[N+T] y_pred;
  vector[T] mu_pred;
  vector[T] sigma_mu_pred;

  sigma_mu_pred[1] = exponential_rng(inv_logit(1. / mu[N]));
  mu_pred[1] = normal_rng(inv_logit(1. / mu[N]), sigma_mu_pred[1]);

  for (t in 1:N) {
    y_pred[t] = binomial_rng(P, inv_logit(mu[t]));
  }

  for (t in 2:T) {
    sigma_mu_pred[t] = exponential_rng(inv_logit(1. / mu_pred[t-1]));
    mu_pred[t] = normal_rng(inv_logit(1. / mu_pred[t-1]), sigma_mu_pred[t]);
  }

  for (t in N+1:N+T) {
    y_pred[t] = binomial_rng(P, inv_logit(mu_pred[t-N]));
  }

}
