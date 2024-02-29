import requests
import urllib
import os
import time
import json
import argparse
# python embedding_client.py --port 8300
parser = argparse.ArgumentParser(description='embedding client')
parser.add_argument('--port',default = 80)
args = parser.parse_args()
url = 'http://localhost:' + str(args.port) + '/generate'
#url = 'http://192.18.78.0:' + str(args.port) + '/generate'
#url = 'http://180.184.91.13:' + str(args.port) + '/generate'
data = dict()
data['user_id'] = 123
data['appid'] = 0
data['timestamp'] = int(time.time() * 1000) #ms 
data['data_list'] = []
data_a = dict()
# data_a['stat'] = 0
# data_a['model_name'] = ''
# data_a['model_version'] = ''
data_a['data'] = "My name is Sarah and I live in London"
# data_a['value'] = 'xxx'
data['data_list'].append(data_a)

data_b = dict()
data_b['data'] = "you are a bad guy"
data['data_list'].append(data_b)
data['size'] = len(data['data_list'])

response = requests.post(url,json=data,stream=True)
print('total cost time: ', (time.time() * 1000 - data['timestamp']))
ans = ''
for c in response.iter_lines():
    js = json.loads(c)
    print(js)
    print("client->server cost time: ", (js['timestamp'] - data['timestamp']))
    
    # ans += js['generated_text']
# print(ans)
# print(response)
