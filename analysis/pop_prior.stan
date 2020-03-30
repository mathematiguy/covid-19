data {
  int N;
  vector[N] X;
  vector[N] Y;
}

generated quantities {
  vector[N] u;

  real beta = normal_rng(1000, 100);
  real<lower=0> s_y = 2;
  real<lower=0> s_u = 2;

  vector[N] y_hat;
  vector[N] y_prior;

  u[1] = lognormal_rng(log(2000000), 1);
  for (t in 2:N)
    u[t] = normal_rng(u[t-1], s_u);

  y_hat = u + beta * X;

  for (t in 1:N)
    y_prior[t] = normal_rng(y_hat[t], s_y);

}
