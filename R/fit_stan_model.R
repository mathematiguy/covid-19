library(here)
library(rstan)
library(optparse)
library(tidyverse)

# Enable multicore
options(mc.cores=4)

main <- function() {

    option_list = list(
        make_option(c("-c", "--case_data"), type="character", default="data/auckland_cases.csv",
                    help="Path to auckland_cases.csv", metavar="character"),
        make_option(c("-m", "--model_code"), type="character", default="stan/sir_model.stan",
                    help="Path to sir_model.stan", metavar="character"),
        make_option(c("-o", "--model_path"), type="character", default="stan/sir_model.rds",
                    help="Path to save model file", metavar="character")
    );

    opt_parser = OptionParser(option_list=option_list);
    opt = parse_args(opt_parser);

    # Load the case data
    message(paste0("Loading the case data from: ", opt$case_data))
    auckland_cases <- read_csv(opt$case_data)

    # Select a subsample of data where it looks like an SIR model would apply
    sir_data <- auckland_cases %>%
        mutate(S = 5000000 - Total,
               I = Active,
               R = Recovered + Deceased) %>%
        select(Date, S, I, R)

    # time series of cases
    cases <- sir_data$I

    # total count
    N <- 5000000;

    # times
    n_days <- as.integer(max(sir_data$Date) - min(sir_data$Date) + 1)
    t <- seq(0, n_days, by = 1)
    t0 = 0
    t <- t[-1]

    #initial conditions
    i0 <- 1
    s0 <- N - i0
    r0 <- sir_data$R[1]
    y0 = c(S = s0, I = i0, R = r0)

    # data for Stan
    data_sir <- list(n_days = n_days, y0 = y0, t0 = t0, ts = t, N = N, cases = cases)

    model_code_path <- here("stan/sir_model.stan")
    message(paste0("Compiling the model at ", model_code_path))
    model <- stan_model(model_code_path)

    message("Fitting the model...")
    fit_sir_negbin <- sampling(
        model,
        data = data_sir,
        iter = 2000,
        chains = 4,
        seed = 0)

    fit_sir_negbin@stanmodel@dso <- new("cxxdso")

    model_path <- here("stan/sir_model.rds")
    message(paste0("Saving the model to: ", opt$model_path))
    saveRDS(fit_sir_negbin, file = opt$model_path)

}

if (!interactive()) {
    main()
    cat("Done!")
}
