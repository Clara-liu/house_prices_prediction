import requests
import json
import os
import logging
import pandas as pd

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm


def strip_invalid_words(region_name: str)-> str:
    stop_words = ['city', 'council', 'borough']
    return filter(lambda x: x not in stop_words, region_name.split())


def format_query(region: str, start_month: str, end_month: str)->str:
    formatted_region = '-'.join(strip_invalid_words(region.lower()))
    values = [formatted_region, start_month, end_month, formatted_region]  # date should follow the format yyyy-mm

    query = """PREFIX  xsd:  <http://www.w3.org/2001/XMLSchema#>
            PREFIX  ukhpi: <http://landregistry.data.gov.uk/def/ukhpi/>

            SELECT  ?item ?ukhpi_refMonth ?ukhpi_refRegion ?ukhpi_averagePriceDetached ?ukhpi_averagePriceFlatMaisonette ?ukhpi_averagePriceSemiDetached ?ukhpi_averagePriceTerraced
            WHERE
            {{ {{ SELECT  ?ukhpi_refMonth ?item
                WHERE
                    {{ ?item  ukhpi:refRegion  <http://landregistry.data.gov.uk/id/region/{}> ;
                            ukhpi:refMonth   ?ukhpi_refMonth
                    FILTER ( ?ukhpi_refMonth >= "{}"^^xsd:gYearMonth )
                    FILTER ( ?ukhpi_refMonth <= "{}"^^xsd:gYearMonth )
                    }}
                ORDER BY ?ukhpi_refMonth
                }}
                OPTIONAL
                {{ ?item  ukhpi:averagePriceDetached  ?ukhpi_averagePriceDetached }}
                OPTIONAL
                {{ ?item  ukhpi:averagePriceFlatMaisonette  ?ukhpi_averagePriceFlatMaisonette }}
                OPTIONAL
                {{ ?item  ukhpi:averagePriceSemiDetached  ?ukhpi_averagePriceSemiDetached }}
                OPTIONAL
                {{ ?item  ukhpi:averagePriceTerraced  ?ukhpi_averagePriceTerraced }}
                BIND(<http://landregistry.data.gov.uk/id/region/{}> AS ?ukhpi_refRegion)
            }}""".format(*values)
    
    return query

def request_data(query: str, region: str)-> dict:
    endpoint = 'https://landregistry.data.gov.uk/landregistry/query'
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    sparql.setQuery(query)

    try:
        ret = sparql.queryAndConvert()['results']['bindings']
    except Exception as e:
        logging.error(f'Encountered error during sparQl request: {e}')
        return None
    
    # list columns of interest
    vars = ['ukhpi_refMonth',
            'ukhpi_averagePriceDetached',
            'ukhpi_averagePriceFlatMaisonette',
            'ukhpi_averagePriceSemiDetached',
            'ukhpi_averagePriceTerraced']
    
    data_formatter = lambda x: {key: value['value'] for key, value in x.items() if key in vars}
    results = list(map(data_formatter, ret))
    df = pd.DataFrame.from_records(results)
    df['Region'] = region
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # find where we are
    abspath = os.path.abspath(__file__)
    absdir = os.path.dirname(abspath)
    os.chdir(absdir)
    # set date range
    start = '2022-01'
    end = '2022-05'
    with open('authorities_list.txt', 'r') as f:
        my_list = [line.rstrip() for line in f]
    saved_data = False
    for region in tqdm(my_list):
        query = format_query(region, start, end)
        df = request_data(query, region)
        if df is None:
            logging.warn(f'Data for {region} not obtained.')
        else:
            if not saved_data:
                df.to_csv('data.txt', index=False, sep = '\t')
                saved_data = True
            else:
                df.to_csv('data.txt', index=False, sep = '\t', header=False, mode='a')
    # count available regions
    data = pd.read_csv('data.txt', sep = '\t')
    available_regions = data['Region'].unique()
    logging.info(f'Out of {len(my_list)}, {len(available_regions)} regions have available data.')