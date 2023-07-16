import requests
from bs4 import BeautifulSoup
import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from typing import List

def get_soup(url: str) -> BeautifulSoup:
    """
    Input:
        url: The URL of the webpage to retrieve.
    Purpose:
        Retrieves the BeautifulSoup object by sending a GET request to the specified URL.
    Returns:
        The BeautifulSoup object representing the parsed HTML content.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup


def get_all_article_links(soup: BeautifulSoup, base_link: str) -> List[str]:
    """
    Input:
        soup: The BeautifulSoup object representing the parsed HTML content.
        base_link: The base URL to be concatenated with the extracted article links.
    Purpose:
        Extracts all article links from the provided BeautifulSoup object.
    Returns:
        A list of all article links extracted from the webpage.
    """
    all_article_link_objects = soup.body.find("div", {"id": "content"}).find("div", {"class": "cleared"}).find("ul", {"class": "archive-articles debate link-box"}).find_all("a", href=True)
    all_article_links = [base_link+link["href"] for link in all_article_link_objects]
    return all_article_links


def find_article_on_page(url: str) -> str:
    """
    Input:
        url: The URL of the webpage containing the article.
    Purpose:
        Retrieves the article text from the provided URL.
    Returns:
        The text content of the article.
    """
    article_soup = get_soup(url)
    article_text = article_soup.body.find("div", {"id": "js-article-text"}).text
    return article_text


def collect_list_of_articles(all_article_links: List[str]) -> List[str]:
    """
    Input:
        all_article_links: A list of article links.
    Purpose:
        Collects the contents of articles by iterating through the provided article links.
    Returns:
        A list of articles.
    """
    all_articles = []
    for article_link in all_article_links:
        article = find_article_on_page(article_link)
        all_articles.append(article)
    return all_articles


def save_list(file_name: str, list_to_dump: List[str]) -> None:
    """
    Input:
        file_name: The name of the file to save.
        list_to_dump: The list to be saved.
    Purpose:
        Saves a list to a file using pickle.
    """
    with open(f"{file_name}.pkl", "wb") as file:
        pickle.dump(list_to_dump, file)


def run_whole_data_pipeline(parent_url: str, base_link: str, file_name: str) -> None:
    """
    Input:
        parent_url: The URL of the parent page.
        base_link: The base link for constructing article links.
        file_name: The name of the file to save the list of articles.
    Purpose:
        Runs the entire data pipeline by retrieving article links, collecting articles,
        and saving the list of articles to a file.
    """
    parent_soup = get_soup(parent_url)
    all_article_links = get_all_article_links(parent_soup, base_link)
    all_articles = collect_list_of_articles(all_article_links)

    save_list(file_name, all_articles)


def get_article_list(path: str) -> List[str]:
    """
    Input:
        path: The path to the file containing the articles.
    Purpose:
        Retrieves a list of articles from a file.
    Returns:
        A list of articles.
    """
    with open(path, "rb") as file:
        articles_list = pickle.load(file)
    return articles_list


def tokenise_articles(articles_list: List[str]) -> List[List[str]]:
    """
    Input:
        articles_list: A list of articles.
    Purpose:
        Tokenises a list of articles.
    Returns:
        A list of tokenised articles.
    """
    tokenised_articles = [article.split() for article in articles_list]
    return tokenised_articles


def transform_tokens_to_vectors(tokenised_articles: List[List[str]], model) -> List[np.ndarray]:
    """
    Input:
        tokenised_articles: A list of tokenised articles.
        model: The model used for vector representation.
    Purpose:
        Transforms tokenised articles into vector representations using a model.
    Returns:
        A list of vector representations of the articles.
    """
    article_vectors = []
    for article in tokenised_articles:
        vectors = [model[word] for word in article if word in model]
        article_vector = np.mean(vectors, axis=0) if vectors else np.zeros(300)
        article_vectors.append(article_vector)
    return article_vectors


def save_vector_representation(file_path: str, article_vectors: List[np.ndarray]) -> None:
    """
    Input:
        file_path: The path to the file for saving the vector representations.
        article_vectors: A list of vector representations of the articles.
    Purpose:
        Saves the vector representations of articles to a file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(article_vectors, file)


def normalise_pos_input(unnormalised_dataset: np.ndarray) -> np.ndarray:
    """
    Input:
        unnormalised_dataset: Array containing the unnormalised input dataset.
    Purpose:
        Normalise and make input dataset positive.
    Returns:
        Array containing the normalised, positive input dataset.
    """
    scaler = MinMaxScaler()
    scaler.fit(unnormalised_dataset)
    normalised_data = scaler.transform(unnormalised_dataset)
    return normalised_data


def run_whole_vectorisation_pipeline(articles_path: str, model, vector_representation_path: str) -> None:
    """
    Input:
        articles_path: The path to the file containing the articles.
        model: The model used for vector representation.
        vector_representation_path: The path to save the vector representations.
    Purpose:
        Runs the entire vectorisation pipeline by loading articles, tokenising them,
        transforming tokens into normalised vector representations, and saving the vector representations.
    """
    articles_list = get_article_list(articles_path)
    tokenised_articles = tokenise_articles(articles_list)
    article_vectors = transform_tokens_to_vectors(tokenised_articles, model)
    reshaped_article_vectors = np.reshape(article_vectors, (len(article_vectors), -1))
    normalised_article_vectors = normalise_pos_input(reshaped_article_vectors)
    save_vector_representation(vector_representation_path, normalised_article_vectors)


def open_list(file_name: str) -> np.ndarray:
    """
    Input:
        file_name: The name of the file to open.
    Purpose:
        Opens a list to a file using pickle.
    """
    with open(f"{file_name}.pkl", "rb") as file:
        articles_list = pickle.load(file)
    return articles_list