functions {

  real softplus(real x) {
    // A numerically stable implementation of the softplus
    // function: log(1 + exp(x))
    if (x > 10)
      return x;
    else if (x < -40)
      return 0;
    else
      return log(1 + exp(x));
  }

}

data {
  int N;
  vector[N] y;
}

parameters {
  real alpha_raw;
  real beta_raw;
  real gamma_raw;
}

transformed parameters {
  real alpha;
  real beta;
  real gamma;

  alpha = 50.0 + 20.0 * alpha_raw;
  beta = 20.0 + 1 * beta_raw;
  gamma = 1 / 5.0 * gamma_raw;

}

model {

  // priors
  alpha_raw ~ std_normal();
  beta_raw ~ std_normal();
  gamma_raw ~ exponential(1);

  // likelihood
  for (i in 1:N)
    y[i] ~ exponential(1 / (gamma + alpha * softplus(i - beta)));

}

generated quantities {

  // Let y be a vector of generated quantities
  // and having y_star a vector of N-1
  // then y1 = beta and y_2:y_N be ystar

  vector[N] y_pred;

  for (i in 1:N)
    y_pred[i] = gamma + alpha * softplus(i - beta);

}
