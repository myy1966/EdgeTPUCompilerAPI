"""
edge_tpu_compiler_api

This module implements the Edge TPU compiler API.
:copyright: (c) 2018 by myy1966.
:license: BSD3, see LICENSE for more details.
"""

import requests
import json
import time
import argparse

api_url = "https://gweb-coral-compiler.appspot.com/api/"

def _download_a_file(download_link, file_name, proxy):
    try:
        r = requests.get(download_link, proxies=proxy)
        save_file = open(file_name, 'wb')
        save_file.write(r.content)
        save_file.close()
    except requests.exceptions.Timeout as _:
        print("Timeout:" + download_link)
    except requests.exceptions.ConnectionError as _:
        print("ConnectionError:" + download_link)
    except requests.exceptions.HTTPError as _:
        print("HTTPError:" + download_link)
    except requests.exceptions.RequestException as _:
        print("RequestException:" + download_link)

def compile(upload_fn, download_fn, proxy=None):
    """compile a tflite model 
    """
    # step 1
    step_1_url = api_url + "compile/upload_uri"
    resp = requests.get(step_1_url, proxies=proxy)

    # step 2
    step_2_url = json.loads(resp.text[resp.text.index('{'):])["url"]
    tflite = {"file1": open(upload_fn, "rb")}
    resp = requests.post(step_2_url, files=tflite, proxies=proxy)
    step_2_resp = json.loads(resp.text[resp.text.index('{'):])

    # step 3
    download_link = step_2_resp["link"]
    step_3_url = api_url + step_2_resp["name"]
    while (True):
        resp = requests.get(step_3_url, proxies=proxy)
        step_3_resp = json.loads(resp.text[resp.text.index('{'):])
        if (step_3_resp["metadata"]["state"] == "SUCCEEDED"):
            print(step_3_resp["log"])
            print("Downloading!")
            _download_a_file(download_link, download_fn, proxy)
            print("Downloaded!")
            return True
        elif (step_3_resp["metadata"]["state"] == "FAILED"):
            print(step_3_resp["log"])
            return False
        elif (step_3_resp["metadata"]["state"] != "IN_PROGRESS"):
            time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("upload_fn")
    parser.add_argument("download_fn")
    parser.add_argument("-p", "--proxy", help="network proxy with both http and https")
    args = parser.parse_args()

    if args.proxy is not None:
        proxies = {
            "http": args.proxy,
            "https": args.proxy,
        }
    else:
        proxies = None
    compile(args.upload_fn, args.download_fn, proxies)

    # proxies = {
    #     "http": "http://127.0.0.1:1087",
    #     "https": "http://127.0.0.1:1087",
    # }
    # compile("mobilenet.tflite", "mobilenet_tpu.tflite", proxies)
