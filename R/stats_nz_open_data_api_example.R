library(here)
library(yaml)

source(here("R/get_odata_fun.R"), local=TRUE)
source(here("R/get_odata_catalogue_fun.R"), local=TRUE)

credentials = read_yaml(here('credentials.yaml'))
service = "https://api.stats.govt.nz/opendata/v1/data.json"

config_proxy <- use_proxy(
    url = curl::ie_get_proxy_for_url(service),
    auth = "any",
    username = ""
)

result <- GET(
    service,
    config_proxy,
    add_headers(.headers = c("Content-Type" = "application/json;charset=UTF-8",
                             "Ocp-Apim-Subscription-Key" = credentials$stats_nz_api_key)),
    timeout(60)) %>%
    content("text", encoding="UTF-8") %>%
    jsonlite::fromJSON(flatten=TRUE)

parsed <- jsonlite::fromJSON(content(result, "text", encoding = "UTF-8"), flatten = TRUE)

# Find available datasets
Catalogue <- get_odata_catalogue(
    service="https://api.stats.govt.nz/opendata/v1",
    endpoint="data.json",
    service_api_key = credentials$stats_nz_api_key
)

# Using the first accessURL from the cataloge, find the entities in the
# service data model
ServiceEntities <-  Filter(
    function(x)!all(is.na(x)),
    get_odata(
         service = "https://api.stats.govt.nz/opendata/v1",
         endpoint = "EmploymentIndicators",
         entity = "",
         query_option = "",
        service_api_key = credentials$stats_nz_api_key
    )
)

# Using the service entity list, get 10 rows for each entity
Observations <-  Filter(
    function(x)!all(is.na(x)),
    get_odata(
        service = "https://api.stats.govt.nz/opendata/v1",
        endpoint = "EmploymentIndicators",
        entity = "Observations",
        query_option = "$select=ResourceID,Period,Duration,Label1,Label2,Value,Unit,Measure,Multiplier&$top=10",
        service_api_key = credentials$stats_nz_api_key)
)

# Example use of filtering:
Observations <-  Filter(
    function(x)!all(is.na(x)),
    get_odata(
        service = "https://api.stats.govt.nz/opendata/v1",
        endpoint = "EmploymentIndicators",
        entity = "Observations",
        query_option = "$filter=(
                                ResourceID eq 'MEI1.1' and
                                Period ge 2020-08-31 and
                                Label2 eq 'Actual' and
                                Duration eq 'P1M'
                              )
                      &$select=ResourceID,Period,Duration,Label1,Label2,Value,Unit,Measure,Multiplier
                      &$top=10",
        service_api_key = credentials$stats_nz_api_key)
)

# Example use of groupby: find all unique combination of values for Label1, Label2 and Measure
Observations <-  Filter(
    function(x)!all(is.na(x)),
    get_odata(
        service = "https://api.stats.govt.nz/opendata/v1",
        endpoint = "EmploymentIndicators",
        entity = "Observations",
        query_option = "$filter=(
                                ResourceID eq 'MEI1.1' and
                                Period ge 2020-08-31 and
                                Duration eq 'P1M'
                              )
                      &$apply=groupby((Label1,Label2,Measure))
                      &$top=10",
        service_api_key = credentials$stats_nz_api_key)
)

# Get the SIR model data
sir_data <-  Filter(
    function(x)!all(is.na(x)),
    get_odata(
        service = "https://api.stats.govt.nz/opendata/v1",
        endpoint = "Covid-19Indicators",
        entity = "Observations",
        query_option = "$filter=(
                     ResourceID eq 'CPCOV2' and
                     Period ge 2021-08-01
                   )
            &$select=Period,Label1,Value",
        service_api_key = credentials$stats_nz_api_key)) %>%
    as_tibble() %>%
    rename(Compartment = Label1,
           Count = Value,
           Day = Period)
