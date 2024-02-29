import requests
import urllib
import os
import time
import json
import argparse
# python translate_client.py --port 8300
parser = argparse.ArgumentParser(description='translation client')
parser.add_argument('--port',default = 9891)
args = parser.parse_args()
#url = 'http://192.18.68.223:' + str(args.port) + '/generate'
url = 'http://180.184.67.112:' + str(args.port) + '/generate'
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
data_a['data_type'] = "prompt"
# data_a['value'] = 'xxx'
data['data_list'].append(data_a)

data_b = dict()
data_b['data'] = "国外华人在一起喊口号让president down台，不知是为何事"
data_b['data_type'] = "prompt"
data['data_list'].append(data_b)
data['size'] = len(data['data_list'])

response = requests.post(url,json=data,stream=True)
ans = ''
for c in response.iter_lines():
    js = json.loads(c)
    print(js)
    # ans += js['generated_text']
    print("client->server cost time: ", (js['timestamp'] - data['timestamp']))
    print('total cost time: ', (time.time() * 1000 - data['timestamp']))
# print(ans)
# print(response)
