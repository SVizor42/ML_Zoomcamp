import pandas as pd
import requests
import click
from time import sleep


@click.command()
@click.option("--host", default="localhost:8000")
@click.option("--dataset_path", default="data/external/rottentomatoes.csv")
def send_requests(host: str, dataset_path: str):
    """
    Generates a list of requests from the file and sends it to the host.
    :param host: hostname:port or URL address
    :param dataset_path: path to the datafile to generate requests from
    :return: nothing
    """
    data = pd.read_csv(dataset_path, sep=';')
    for request_data in data.to_dict('records'):
        print(f'Request to service: {request_data}')
        response = requests.post(url=f'http://{host}/predict/', json=request_data)
        print(f'Status code: {response.status_code}')
        print(f'Result: {response.text}')
        sleep(1)


if __name__ == "__main__":
    send_requests()
