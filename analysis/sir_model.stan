// SIR model by Arie Voorman.
// Code from here: https://rstudio-pubs-static.s3.amazonaws.com/270496_e28d8aaa285042f2be0c24fc915a68b2.html

data {

  //declare variables
  int<lower=0> N; //number of individuals
  int<lower=0> T; //number of time points
  int<lower = 0, upper =N> S[T]; //SIR
  int<lower = 0, upper =N> I[T];
  int<lower = 0, upper =N> R[T];
  real<lower=0> lambda_max; //maximum value of lambda (sometimes useful to help STAN converge)

}
transformed data {

  //Calculate transitions, based on SIR status
  int<lower=0, upper = N> z_si[T-1];
  int<lower=0, upper = N> z_ir[T-1];

  for(i in 1:(T-1)){
    z_si[i] = S[i] - S[i+1];
    z_ir[i] = R[i+1] - R[i];
  }

}
parameters {
  real<lower=1> gamma;
  real<lower=0, upper = lambda_max> lambda;
}
model {

  lambda ~ uniform(0,lambda_max);

  for(i in 1:(T-1)){
     if(I[i] > 0){ //only define z_si when there are infections - otherwise distribution is degenerate and STAN has trouble
       z_si[i] ~ binomial(S[i], 1-(1-lambda)^I[i]);
     }
     z_ir[i] ~ binomial(I[i], 1/gamma);
  }
}

generated quantities {

  int<lower=0> r_0; // simulate a single infected individual in the population, and count secondary infections (i.e. R_0).
  r_0 = 0;

  while(1){
    r_0 = r_0 + binomial_rng(N-r_0,lambda);
    if(bernoulli_rng(1/gamma)) break;
  }

}
