from dataproc import DataScraper, Dataflow
import json

json_path = 'dataproc/dataset/'
json_name = 'dataset.json'

DataScraper.dataScraper(
    pool=['Li', 'Ni', 'Co', 'O', 'Mn'],
    api_key='dy2KrftCVWbLNIFIG56n',
    json_path=json_path,
    json_name=json_name,
    sources=['xrd.Cu', 'xrd.Mo', 'xrd.Fe', 'xrd.Ag']
)


