import requests
from bs4 import BeautifulSoup


def gettingData():
    # Define the URL
    url = 'https://raw.githubusercontent.com/spyguessgame-boop/own_dataset/refs/heads/main/data.txt'

    # Fetch the content from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all text from the page
    text_data = soup.get_text()

    # Print the first 1000 characters of the extracted text to verify
    print(text_data[:10000])
    return text_data