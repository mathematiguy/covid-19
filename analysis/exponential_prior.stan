data {
  int N;
  vector[N] y;
}

generated quantities {

  real alpha;
  vector[N] y_prior;

  alpha = normal_rng(0.2, 0.04);

  // likelihood
  y_prior[1] = normal_rng(1, 0.1);
  for (i in 2:N)
    y_prior[i] = (1 + alpha) * y_prior[i-1];

}
