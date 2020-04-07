data {

  //declare variables
  int<lower=0> N; //number of individuals
  int<lower=0> T; //number of time points
  int<lower = 0, upper =N> S[T]; // Susceptible
  int<lower = 0, upper =N> E[T]; // Exposed
  int<lower = 0, upper =N> P[T]; // Pre-symptomatic
  int<lower = 0, upper =N> I[T]; // Infectious
  int<lower = 0, upper =N> R[T]; // Recovered

}

generated quantities {

  real<lower=0> R0;     // The basic reproduction number

  R0 = exponential_rng(1);


}
