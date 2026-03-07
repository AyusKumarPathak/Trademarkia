import os
import re

# Path where the extracted dataset exists
DATASET_PATH = "dataset/20_newsgroups"


def clean_text(text: str) -> str:
    """
    Clean noisy Usenet message data.

    The 20 Newsgroups dataset contains:
    - email headers
    - quoted replies
    - PGP signatures
    - email addresses
    - network metadata

    These elements do NOT represent the semantic topic of the document
    and negatively affect embedding quality.

    We remove them while preserving natural language structure so that
    transformer embedding models can capture semantic meaning.
    """

    # Remove header block (everything before the first blank line)
    text = re.split(r'\n\s*\n', text, maxsplit=1)[-1]

    # Remove quoted replies (lines starting with ">")
    text = re.sub(r'^>.*$', '', text, flags=re.MULTILINE)

    # Remove PGP signatures
    text = re.sub(r'-----BEGIN PGP.*?END PGP SIGNATURE-----', '', text, flags=re.S)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove special characters but keep words and numbers
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.lower().strip()


def load_dataset():
    """
    Load dataset from directory structure.

    Each folder represents a topic category.
    Each file represents a document.

    Returns
    -------
    documents : list[str]
    labels : list[str]
    """

    documents = []
    labels = []

    for category in os.listdir(DATASET_PATH):

        category_path = os.path.join(DATASET_PATH, category)

        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            try:
                with open(file_path, "r", encoding="latin1") as f:

                    raw_text = f.read()

                    cleaned = clean_text(raw_text)

                    # Filter extremely short documents
                    # Very short texts produce unstable embeddings
                    if len(cleaned.split()) > 20:
                        documents.append(cleaned)
                        labels.append(category)

            except:
                continue

    return documents, labels


if __name__ == "__main__":

    docs, labels = load_dataset()

    print("Documents loaded:", len(docs))
    print("Unique categories:", len(set(labels)))
    print("\nExample document:\n")
    print(docs[0][:400])