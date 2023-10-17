## Requriment

```
pip install -r requirements.txt
```
## Make some changes in site package
### Step 1 :
```
cd /home/ubuntu/.local/lib/python3.10/site-packages/fedml/core/distributed/crypto
```
### Step 2 :

```
nano cypto_api.py
```
### Step 3 :

 Comment line 5 as 
```
# from ecies.utils import aes_decrypt, aes_encrypt
```

save it

## Training Script

At the client side, the client ID (a.k.a rank) starts from 1.
Please also modify config/fedml_config.yaml, changing the `worker_num` the as the number of clients you plan to run.

If you want to use your customized MQTT or web3 storage server as training backends, you should uncomment and set the following lines.
#customized_training_mqtt_config: {'BROKER_HOST': 'your mqtt server address or domain name', 'MQTT_PWD': 'your mqtt password', 'BROKER_PORT': 1883, 'MQTT_KEEPALIVE': 180, 'MQTT_USER': 'your mqtt user'}
#customized_training_web3_config: {'token': 'your ipfs token at web3.storage', 'upload_uri': 'https://api.web3.storage/upload', 'download_uri': 'ipfs.w3s.link'}

Use Web3 storage as federated learning distributed storage for reading and writing models.
You should register account at https://web3.storage and set config parameters:
token, upload_uri, download_uri.
If you want to use secret key to encrypt models, you should set secret key by calling Context().add("ipfs_secret_key", "your secret key")

At the server side, run the following script:
```
bash run_server.sh 
```

For client 1, run the following script:
```
bash run_client_1.sh 1 your_run_id
```
For client 2, run the following script:
```
bash run_client_2.sh 2 your_run_id
```
Note: please run the server first.

