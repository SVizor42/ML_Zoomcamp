import json
import yaml
import re
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def read_config(config_path: str) -> dict:
    """
    Reads config from .yaml file.
    :param config_path: path to .yaml file to load from
    :return: configuration dictionary
    """
    with open(config_path, "rb") as f_in:
        config = yaml.safe_load(f_in)
    return config


def parse_config(config: dict) -> dict:
    """
    Parses the configuration dictionary.
    :param config: configuration dictionary
    :return: configuration parameters
    """
    params = dict()
    params['input_data_path'] = config['input_data_path']
    params['output_model_path'] = config['output_model_path']
    params['metric_path'] = config['metric_path']
    params['split_val_size'] = config['splitting_params']['val_size']
    params['split_random_state'] = config['splitting_params']['random_state']
    params['transformer_type'] = config['train_params']['transformer_type']
    params['transformer_max_df'] = config['train_params']['transformer_max_df']
    params['transformer_max_features'] = config['train_params']['transformer_max_features']
    params['transformer_ngram_range'] = (
        int(config['train_params']['transformer_min_ngram']),
        int(config['train_params']['transformer_max_ngram'])
    )
    params['model_type'] = config['train_params']['model_type']
    params['model_random_state'] = config['train_params']['model_random_state']
    params['model_C'] = config['train_params']['model_C']

    return params


def remove_html_tags(text: str) -> str:
    """
    Throws away HTML tags from text.
    :param text: text to process
    :return: text without HTML tags
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_emoji(text: str) -> str:
    """
    Removes all special symbols from the text.
    :param text: text to process
    :return: text without special symbols
    """
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_urls(text: str) -> str:
    """
    Removes URL's from the text.
    :param text: text to process
    :return: text with no URLs
    """
    return re.sub(r'http\S+', '', text)


def remove_punctuations(text: str) -> str:
    """
    Removes punctuation signs.
    :param text: text to process
    :return: text with no punctuations
    """
    return ''.join(
        [word.lower() for word in text if word not in string.punctuation]
    )


def remove_stopwords(text: str) -> str:
    """
    Removes stopwords.
    :param text: text to process
    :return: text after stopwords deleting
    """
    stop_words = stopwords.words('english')
    return ' '.join(
        [word for word in text.split() if word not in stop_words and word.isalpha()]
    )


def add_space(text: str) -> str:
    """
    Adds space after dot symbol.
    :param text: initial text
    :return: processed text
    """
    return re.sub(r'\.(?=\S)', '. ', text)


def process_mappings(text: str) -> str:
    """
    Replaces word contractions with their original form.
    :param text: initial text
    :return: processed text
    """
    with open('data/external/mappings.json') as json_file:
        mappings = json.load(json_file)
    return ' '.join(
        [mappings[word] if word in mappings else word for word in text.split(' ')]
    )


def apply_lemmatizer(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """
    Applies WordNetLemmatizer to the text.
    :param text: text to process
    :param lemmatizer: WordNetLemmatizer
    :return: lemmatized text
    """
    return ' '.join(
        [lemmatizer.lemmatize(word) for word in text.split() if word.isalpha()]
    )


def process_text(text: str, lemmatize: bool = True) -> str:
    """
    Executes text processing: restores contractions to their originals, then removes HTML tags,
    URLs, stopwords, punctuations and other special chars; finally, applies the lemmatizer.
    :param text: text to process
    :param lemmatize: flag of using lemmatizer
    :return: processed text
    """
    text = remove_html_tags(text)
    text = process_mappings(text)
    text = remove_emoji(text)
    text = add_space(text)
    text = remove_urls(text)
    text = remove_punctuations(text)
    text = remove_stopwords(text)

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        text = apply_lemmatizer(text, lemmatizer)

    return text
