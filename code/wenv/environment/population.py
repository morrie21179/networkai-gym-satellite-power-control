import requests
import json
import qwikidata
import qwikidata.sparql
import time

"""
Obtain population density of city via OpenDataSoft API
[view table](https://public.opendatasoft.com/explore/dataset/geonames-all-cities-with-a-population-1000/table)
(hardly have any pop. data in OpenStreetMap)
"""
def get_city_opendata(city):
    url = f'https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/geonames-all-cities-with-a-population-1000/records?where=cou_name_en%20%3D%20%22{city}%22&offset=0&timezone=UTC&include_links=false&include_app_metas=false'
    res = requests.get(url)
    # parse response
    dct = json.loads(res.content)
    total_count = dct["total_count"]
    print("total count:", total_count)
    
    # save json
    saveFile = f"{city}_data_{total_count}.json"
    out = dct['results']
    with open(saveFile, 'w') as f:
        json.dump(out, f)

    return out

"""
Obtain population of city via QWikiData API
https://query.wikidata.org/
"""
def get_city_wikidata(city:str, country:str):
    query = """
    SELECT ?city ?cityLabel ?country ?countryLabel ?population
    WHERE
    {
      ?city rdfs:label '%s'@en.
      ?city wdt:P1082 ?population.
      ?city wdt:P17 ?country.
      ?city rdfs:label ?cityLabel.
      ?country rdfs:label ?countryLabel.
      FILTER(LANG(?cityLabel) = "en").
      FILTER(LANG(?countryLabel) = "en").
      FILTER(CONTAINS(?countryLabel, "%s")).
    }
    """ % (city, country)

    try:
        res = qwikidata.sparql.return_sparql_query_results(query)
    except requests.exceptions.JSONDecodeError:
        print("*decode error.. sleep and reconnecting")
        time.sleep(3)
        return get_city_wikidata(city, country)

    if (len(res['results']['bindings']) > 0):
        out = res['results']['bindings'][0]
    else:
        out = None

    return out

# ASCII Name == Ivanivka
if __name__ == '__main__':
    city_data = get_city_wikidata(city="Sumy Oblast", country='Ukraine')
    print(city_data["cityLabel"]["value"], ':', city_data["population"]["value"])