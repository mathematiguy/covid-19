data {
  int N;
  vector[N] y;
}

parameters {
  real<lower=0> alpha_raw;
  real<lower=0> beta_raw;
}

transformed parameters {
  real alpha;
  real beta;

  alpha = 0.2 + 0.04 * alpha_raw;
  beta = 1 + 0.1 * beta_raw;

}

model {

  // priors
  alpha_raw ~ std_normal();
  beta_raw ~ std_normal();

  // likelihood
  y[1] ~ normal(beta_raw, 1);
  for (i in 2:N)
    y[i] ~ exponential(1 / ((1 + alpha_raw) *  y[i-1]));

}

generated quantities {

  vector[N] y_pred;

  y_pred[1] = beta;
  for (i in 2:N)
    y_pred[i] = (1 + alpha) * y_pred[i-1];

}
