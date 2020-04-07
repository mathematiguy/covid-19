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
  int N;                          // Number of timesteps
}

generated quantities {

  // Model log(1 + exp(ax - b))

  real alpha;                  // Slope
  real<lower=0> beta;          // Location of the kink
  real<lower=0> gamma;         // Level
  vector<lower=0>[N] y_prior;  // Number of cases

  alpha = normal_rng(50, 20);
  beta  = normal_rng(20, 1);
  gamma = exponential_rng(1.0 / 5.0);

  // likelihood
  for (i in 1:N)
    // We use log(1+exp(-abs(x))) + max(x,0) for numerical stability
    y_prior[i] = gamma + alpha * softplus(i - beta);

}
