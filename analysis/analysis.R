library(tidyverse)
library(nCov2019)
library(wbstats)
library(furrr)
library(rstan)
library(here)

# Use furr multiprocessing
plan(multiprocess)

# Set default ggplot theme
theme_set(theme_minimal())

# Set up stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# Get global COVID-19 data
covid_data <- load_nCov2019(lang = 'en')['global'] %>%
    filter(!is.na(country)) %>%
    group_by(time, country) %>%
    summarise(cum_confirm = sum(cum_confirm),
              cum_heal = sum(cum_heal),
              cum_dead = sum(cum_dead))

# Get global population data
pop_data <- wb(indicator = "SP.POP.TOTL", startdate = 1900, enddate = 2019) %>%
    mutate_at(c("date", "value"), as.integer) %>%
    mutate(country = case_when(
               country == "Russian Federation" ~ "Russia",
               country == "Iran, Islamic Rep." ~ "Iran",
               country == "Korea, Dem. People’s Rep." ~ "South Korea",
               country == "Egypt, Arab Rep." ~ "Egypt",
               country == "Brunei Darussalam" ~ "Brunei",
               country == "Venezuela, RB" ~ "Venezuela",
               country == "Bahamas, The" ~ "Bahamas",
               country == "Gambia, The" ~ "Gambia",
               TRUE ~ country)) %>%
    select(-indicator, -indicatorID) %>%
    filter(country %in% covid_data$country) %>%
    arrange(country, date)

pop_vars <- wbsearch("population ages") %>%
    filter(str_detect(indicator, "Population ages"))

density_data <- wb(indicator = "EN.POP.DNST", startdate = 1900, enddate = 2019) %>%
    mutate(date = as.integer(date),
           value = as.numeric(value)) %>%
    mutate(country = case_when(
           country == "Russian Federation" ~ "Russia",
           country == "Iran, Islamic Rep." ~ "Iran",
           country == "Korea, Dem. People’s Rep." ~ "South Korea",
           country == "Egypt, Arab Rep." ~ "Egypt",
           country == "Brunei Darussalam" ~ "Brunei",
           country == "Venezuela, RB" ~ "Venezuela",
           country == "Bahamas, The" ~ "Bahamas",
           country == "Gambia, The" ~ "Gambia",
           TRUE ~ country)) %>%
    select(-indicator, -indicatorID) %>%
    filter(country %in% covid_data$country) %>%
    arrange(country, date)

density_data %>%
    ggplot(aes(x = date, y = value, colour = country)) +
    geom_line() +
    facet_wrap(~country, scales = 'free_y') +
    guides(colour = FALSE)


# Get global GDP data
gdp_data <- wb(indicator = "NY.GDP.MKTP.CD", startdate = 1990, enddate = 2016)

covid_data <- covid_data %>%
    filter(country %in% pop_data$country) %>%
    arrange(country, time)

# Plot population by country
pop_data %>%
    ggplot(aes(x = date, y = value, colour = country)) +
    geom_point() +
    facet_wrap(~country, scales = 'free_y') +
    guides(colour = FALSE) +
    # scale_y_log10() +
    theme_minimal()

pop_data %>%
    group_by(country) %>%
    summarise(endpoint_mean = mean(value[1], value[length(value)]),
              midpoint_value = value[floor(length(value) / 2)])

# Plot covid-19 cases by country
covid_data %>%
    ggplot(aes(x = time, y = cum_confirm, colour = country)) +
    geom_line() +
    facet_wrap(~country, scales = 'free_y') +
    guides(colour = FALSE) +
    theme_minimal()

# Lets start by forecasting global populations
head(pop_data)

pop_data %>%
    arrange(country, date) %>%
    group_by(country) %>%
    do(head(., 1)) %>%
    .$value %>%
    log() %>%
    hist()

nz_pop <- filter(pop_data, country == "New Zealand")

nz_pop %>%
    ggplot(aes(x = date, y = value)) +
    geom_line() +
    geom_point()

mean(nz_pop$value)

# Prior simulation
prior_fit <- stan(
    file = here('analysis/pop_prior.stan'),
    data = list(
        N = nrow(nz_pop),
        X = nz_pop$date,
        Y = nz_pop$value),
    iter = 2000,
    algorithm = "Fixed_param",
    chains = 1
)

y_prior <- extract(prior_fit)$y_prior %>%
     as_tibble() %>%
     add_column(sim = 1:nrow(.), .before = 'V1') %>%
     gather(paste0("V", 1:59), key = 'timestep', value = 'value') %>%
     mutate(timestep = as.integer(str_remove_all(timestep, "V")) + min(nz_pop$date) - 1)

y_prior %>%
    ggplot(aes(x = timestep, y = value, group = sim)) +
    scale_y_log10() +
    geom_line(alpha = 0.1)

# Posterior simulation
pop_fit <- stan(
    file = here('analysis/pop_growth.stan'),
    data = list(
        N = nrow(nz_pop),
        X = nz_pop$date,
        Y = nz_pop$value),
    iter = 20000,
    )

y_hat <- extract(pop_fit)$y_hat %>%
     as_tibble() %>%
     add_column(sim = 1:nrow(.), .before = 'V1') %>%
     gather(paste0("V", 1:59), key = 'timestep', value = 'value')

y_hat %>%
    mutate(timestep = as.integer(str_remove_all(timestep, "V")) + min(nz_pop$date) - 1) %>%
    ggplot(aes(x = timestep, y = value, group = sim)) +
    geom_line(alpha = 0.01)
