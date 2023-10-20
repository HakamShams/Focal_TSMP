# ------------------------------------------------------------
# Download TSMP dataset from Forschungszentrum Jülich GmbH
# ------------------------------------------------------------

import time
import os
import requests
from tqdm import tqdm
import argparse
from bs4 import BeautifulSoup

# ------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, default='../data/TerrSysMP',
                        help='directory to save the dataset [default: ../data/TerrSysMP]')
    parser.add_argument('--url', type=str, default='https://datapub.fz-juelich.de/slts/cordex/data/',
                        help='juelich repository [default: https://datapub.fz-juelich.de/slts/cordex/data/')
    args = parser.parse_args()

    return args


def download_TerrSysMP():

    args = parse_args()
    out_path = args.out_path
    url = args.url
    # create local directory to store data files
    os.makedirs(out_path, exist_ok=True)

    # get internal urls
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    dirs_urls = []
    for link in soup.find_all('a'):
        dirs_urls.append(url + link.get('href'))
    dirs_urls = dirs_urls[-56:]

    # download data from web server
    print("Downloading TerrSysMP Dataset...")
    time.sleep(1)

    for dir_link in dirs_urls:

        reqs = requests.get(dir_link)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        name = os.path.basename(os.path.normpath(dir_link))
        dir_name = os.path.join(out_path, name)
        os.makedirs(dir_name, exist_ok=True)

        data_urls = []
        for link in soup.find_all('a'):
            data_urls.append(link.get('href'))

        data_urls = [data_url for data_url in data_urls if data_url.endswith('nc')]

        pbar = tqdm(enumerate(data_urls), total=len(data_urls), smoothing=0.9, leave=False)
        for i, file in pbar:

            response = requests.get(dir_link + file)
            size = int(response.headers['content-length']) / (1024 * 1024)
            data = os.path.splitext(file)[0][os.path.splitext(file)[0].rfind('_')+1:]

            pbar.set_description('%s %s %.3f MB' % (name, data, size), refresh=True)

            with open(f"{dir_name}/{file}", "wb") as f:
                f.write(response.content)


if __name__ == '__main__':

    download_TerrSysMP()

