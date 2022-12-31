import pandas as pd
import requests
import lxml.html


def load_text_file(filename):
    """Loads a text file and returns the contents as a string."""
    with open(filename, "r") as f:
        return f.read()


def load_csv_file(filename):
    """Loads a CSV file and returns the contents as a Pandas DataFrame."""
    return pd.read_csv(filename)


def load_webpage(url):
    """Loads a webpage and returns the contents as an HTML string."""
    response = requests.get(url)
    return response.text


def parse_html(html):
    """Parses an HTML string and returns a list of text nodes."""
    doc = lxml.html.fromstring(html)
    return doc.xpath("//text()")


def preprocess_text(text):
    """Preprocesses a string of text by lowercasing it and removing punctuation."""
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    return text
