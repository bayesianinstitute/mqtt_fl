#!/usr/bin/env bash
RANK=1
RUN_ID=1
python3 torch_client.py --cf config/fedml_config.yaml --rank $RANK --role client --run_id $RUN_ID 1 11