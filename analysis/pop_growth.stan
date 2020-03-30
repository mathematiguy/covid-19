data {
  int N;
  vector[N] X;
  vector[N] Y;
}

parameters {
  vector[N] u;
  real beta;
  real<lower=0> s_y;
  real<lower=0> s_u;
}

transformed parameters {
  vector[N] y_hat;
  y_hat = u + beta * X;
}

model {
  u[2:N] ~ normal(u[1:(N-1)], s_u);
  Y ~ normal(y_hat, s_y);
}
