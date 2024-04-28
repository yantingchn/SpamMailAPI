import re
import pandas as pd
import seaborn as sns
from typing import Dict, List

# Natural Language Processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords

from pydantic import BaseModel
import yaml

ROOT_PATH = './'

class Config(BaseModel):
    """ config: load config file """
    path : Dict[str, str]
    url : Dict[str, str]

def load_config():
    """ load_config: load config file """
    with open(ROOT_PATH + 'configs/config.yaml') as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)

class ModelConfig(BaseModel):
    """ config: load config file """
    encoded_label: List[str]
    deployed_model : Dict[str, str]

def load_model_config():
    """ load_model_config: load config file """
    with open(ROOT_PATH + 'configs/model.yaml') as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict)

class Loader():
    """ Loader: load dataset and config file """
    def __init__(self):
        self.config = load_config()    

    def load_df(self) -> pd.DataFrame:
        return pd.read_csv(ROOT_PATH + self.config.path['data_path'])

class DataPreprocessing():
    """ DataProcessing: Text cleaning and tokenization with NLTK """

    # Initialise the lemmatizer and stemmer
    def __init__(self):
        self.lemm = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.stop_words = stopwords.words('english')

    # Remove punctuation and tokenize the text, and remove the "subject" tag
    def tokenization(self, line: str) -> list:
        tokens = re.sub(r"[^a-zA-Z0-9]", " ", line.lower()).split()
        return tokens if (not tokens or tokens[0] != "subject") else tokens[1:]

    # Tokenize the text with NLTK, and remove the "subject" tag
    def tokenization_nltk(self, line: str) -> list:
        line = word_tokenize(line)
        line = line if (not line or line[0].lower != "subject") else line[1:]
        line = [self.lemm.lemmatize(word.lower()) for word in line if word.isalpha()]
        line = [self.ps.stem(word) for word in line]

        return line

class Visualisation():
    """ Visualisation: Data visualisation """
    def __init__(self):
        None

    def ClaimCountBarChart(self, df: pd.DataFrame, target) -> None:
        
        total = float(len(df))

        ax = sns.countplot(data=df, x=target, hue=target)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}'.format((height/total)*100),
                    ha="center")
            ax.set_title(target)
