library(httr)
library(here)
library(yaml)
library(optparse)
library(tidyverse)

source(here('R/get_odata_fun.R'), local=TRUE)

main <- function() {

    option_list = list(
        make_option(c("-d", "--case_data"), type="character", default="data",
                    help="Path to cases_by_DHB_over_time.csv", metavar="character"),
        make_option(c("-o", "--output"), type="character", default="data",
                    help="Path to save output auckland_cases.csv", metavar="character"),
        make_option(c("-c", "--credentials"), type="character", default="out.txt",
                    help="Path to credentials file", metavar="character")
    );

    opt_parser = OptionParser(option_list=option_list);
    opt = parse_args(opt_parser);

    # Load API credentials
    credentials = read_yaml(here('credentials.yaml'))

    # Load the auckland case data from the nz-covid19-data-auto github repo
    cases_by_dhb_over_time <- read_csv(here(opt$case_data))

    auckland_cases <- cases_by_dhb_over_time %>%
        filter(DHB=='Auckland' & Date > as.Date('2021-08-18')) %>%
        select(-DHB)

    # Get the testing data from the Stats NZ open data api
    testing_data <-  Filter(
        function(x)!all(is.na(x)),
        get_odata(
            service = "https://api.stats.govt.nz/opendata/v1",
            endpoint = "Covid-19Indicators",
            entity = "Observations",
            query_option = "$filter=(
                         ResourceID eq 'CPCOV1' and
                         Period ge 2021-08-18
                       )
                &$select=Period,Label1,Value",
            service_api_key = credentials$stats_nz_api_key)) %>%
        as_tibble() %>%
        rename(Tests = Value,
               Date = Period) %>%
        mutate(Date = as.Date(Date))

    # Join the case data with the testing data
    auckland_cases <- auckland_cases %>%
        left_join(testing_data)

    # Write the case data to disk
    write_csv(auckland_cases, here(opt$output))

}

if (!interactive()) {
    main()
    cat("Done!")
}
