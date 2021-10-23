source(here('R/get_odata_fun.R'), local=TRUE)

indicators = get_odata(
    service = "https://api.stats.govt.nz/opendata/v1",
    endpoint = "Covid-19Indicators",
    entity = "Observations",
    query_option = "filter=(ResourceID%20eq%20'CPCOV5.0')&$top=10",
    service_api_key = ""
)

