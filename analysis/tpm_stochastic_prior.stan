/*
  Author: Caleb Moses

  Useful notes here:
  - Discrete SIR model https://mathinsight.org/discrete_sir_infectious_disease_model
  - Wikipedia https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology
*/
data {
  //declare variables
  int<lower=0> N; //number of individuals
  int<lower=0> T; //number of time points
}
generated quantities {
  int<lower=0, upper=N> S[T]; //SIR
  int<lower=0, upper=N> I[T];
  int<lower=0, upper=N> R[T];

  int<lower=0, upper=N> n_infected;
  int<lower=0, upper=N> n_recovered;

  real<lower=0, upper=1> a;
  real<lower=0, upper=1> b;

  a = 1.;
  b = 1.;

  I[1] = neg_binomial_rng(10, 0.5);
  S[1] = N - I[1];
  R[1] = 0;

  for (t in 1:T-1) {

    n_infected = binomial_rng(S[t], b * I[t] / N);
    n_recovered = binomial_rng(I[t], a);

    S[t+1] = S[t] - n_infected;
    I[t+1] = I[t] + n_infected - n_recovered;
    R[t+1] = R[t] + n_recovered;

  }

}
