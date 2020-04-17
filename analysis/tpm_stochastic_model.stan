/*
  Author: Caleb Moses

  Useful notes here:
  - Discrete SIR model https://mathinsight.org/discrete_sir_infectious_disease_model
  - Wikipedia https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
*/
data {
  //declare variables
  int<lower=0> N;             // Number of individuals
  int<lower=0> T;             // Number of time points
  int<lower=0, upper=N> S[T]; // Susceptible
  int<lower=0, upper=N> I[T]; // Infected
  int<lower=0, upper=N> R[T]; // Removed
}

parameters {
  real<lower=0> a;
  real<lower=0> b;

  real<lower=0,upper=N> n_infected[T];
  real<lower=0,upper=N> n_recovered[T];
}

model {

  a ~ normal(1., 0.1);
  b ~ normal(1., 0.1);

}

generated quantities {

  I[1] = neg_binomial_rng(10, 0.5);
  S[1] = N - I[1];
  R[1] = 0;

  for (t in 1:T-1) {

    n_infected  = binomial_rng(S[t], b * I[t] / N);
    n_recovered = binomial_rng(I[t], a);

    S[t+1] = S[t] - n_infected;
    I[t+1] = I[t] + n_infected - n_recovered;
    R[t+1] = R[t] + n_recovered;

  }

}
