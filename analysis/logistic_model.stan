data {
  int N;
  vector[N] y;
  int<lower=1> K;
}

parameters {
  real<lower=0> alpha_raw;
  real<lower=0> beta_raw;
}

transformed parameters {
  real<lower=1> alpha;
  real<lower=1> beta;

  alpha = 1.2 + 0.04 * alpha_raw;
  beta = 1 + 0.1 * beta_raw;

}

model {

  // priors
  alpha_raw ~ std_normal();
  beta_raw ~ std_normal();

  // likelihood
  y[1] ~ normal(beta_raw, 1);
  for (i in 2:N)
    y[i] ~ exponential(1 / (y[i-1] + alpha_raw * (1 - y[i-1] / K)));

}

generated quantities {

  vector[N] y_pred;

  y_pred[1] = beta;
  for (i in 2:N)
    y_pred[i] = alpha * y_pred[i-1] * (1 - y[i-1] / K);

}
