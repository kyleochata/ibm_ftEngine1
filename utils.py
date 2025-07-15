import requests
import matplotlib.pyplot as plt
from pathlib import Path

def download(url: str, path: str) -> None:
    """Downloads a file from a URL and saves it locally.

    Args:
        url (str): The URL of the file to download.
        path (str): Local filesystem path where the file will be saved.

    Raises:
        requests.exceptions.RequestException: If the download fails (e.g., network error).
    """
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
    return

def savePlot(path: str) -> None:
    """Saves the current matplotlib figure to a file and closes the plot.

    Args:
        path (str): Output file path (e.g., 'plot.png').
                   Supported formats: PNG, JPG, PDF, SVG, etc.
    """
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return

def checkIfAssetExists(path: str) -> bool: 
    """Checks if a plot file exists on the filesystem.

    Args:
        path (str): Path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(path).exists()