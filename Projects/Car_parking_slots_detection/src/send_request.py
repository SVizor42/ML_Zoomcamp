import requests
import click
import os


@click.command()
@click.option("--host", default="localhost:8000")
@click.option("--dataset_path", default="data/external")
def send_requests(host: str, dataset_path: str):
    """
    Generates requests using images from the dir and sends it to the host.
    :param host: hostname:port or URL address
    :param dataset_path: path to the directory with the test images
    :return: nothing
    """
    for filename in os.listdir(dataset_path):
        file = {'file': (filename, open(os.path.join(dataset_path, filename), 'rb'), 'image/jpeg')}
        response = requests.post(url=f'http://{host}/predict/', files=file)
        print(f'Status code: {response.status_code}')
        print(f'Result: {response.text}')


if __name__ == "__main__":
    send_requests()
