import apache_beam as beam
import logging
import requests
import time
import os

class callAPI(beam.DoFn):
    def __init__(self,header,url,retries):
        self.header=header
        self.url=url
        self.max_retries=retries
    def process(self,element):
        url=self.url
        retries=0
        while retries<self.max_retries:
            res=requests.get(self.url,headers=self.header)
            if res.status_code == 429:
                retry_after = int(res.headers.get("Retry-After", 1))
                logging.warning(f"Rate limited. Sleeping {retry_after}s")
                time.sleep(retry_after)
                retries += 1
                continue
            if not res.ok:
                res.raise_for_status()
                logging.error(res.text)
                return
            yield res.json()
            return

def get_per_group_match_count(element):
    group,matches=element
    return f'{group},{len(matches)}'

with beam.Pipeline() as p:
    token=os.environ.get('TOKEN_FOR_FOOTBALL_API')
    headers={
        'X-Auth-Token':f"{token}",
        'Accept-Encoding': '' 
    }
    url=os.environ.get('URI_FOR_FOOTBALL_API')
    data=(
          p
          | "Seed" >> beam.Create([None])
          | 'Call API' >> beam.ParDo(callAPI(headers,url,5))
          | 'Flatten'>>beam.FlatMap(lambda x:x['matches'])
          | 'Aggregate Count of matches'>>beam.GroupBy(lambda s:s['group'])
          | 'Per Match count per group'>> beam.Map(get_per_group_match_count)
          | 'Write To Txt'>>beam.io.WriteToText('output.txt',shard_name_template='')
    )



# To run code, use the following command:python Get_Fifa_World_Cup_Data.py --runner=DirectRunner
# Will pull the data from https://www.football-data.org/client/home, flatten it and groupby the groups in the world cup 2026, and gets the count of matches in each group, and then writes it to an txt file.
