apt-get update
apt-get install  apache2-utils
#注意post_data中json的'需要替换成"
ab  -T application/json  -n 1000 -c 100 -p post_data http://localhost:80/generate