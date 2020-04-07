data {
  int N;                          // Number of timesteps
  int<lower=1> K;                 // Carrying capacity
}

generated quantities {

  real<lower=1> alpha;            // Maximum growth rate
  vector<lower=1>[N] y_prior;     // Number of cases

  alpha = normal_rng(1.2, 0.04);

  // likelihood
  y_prior[1] = normal_rng(1.2, 0.04);
  for (i in 2:N)
    y_prior[i] = alpha * y_prior[i-1] * (1 - y_prior[i-1] / K);

}
