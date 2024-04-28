# Spam Check API
The Spam Check API is designed to analyze SMS messages and return a spam classification. This document provides a guide on how to set up and use the API.

## Python Environment
> python: 3.8.18 
> build_dependencies:
> - pip==23.3.1 
> - setuptools==68.2.2 
> - wheel==0.41.2

## Installation
```pip install -r requirements.txt```

## Folder Structure
    .
    ├── configs                 # Configuration files
    │   ├── config.yaml         # Misc config (url... etc.)
    │   └── model.yaml          # Deployed Model config
    ├── model_training.ipynb    # Main model traing notebook
    ├── model.py                # Model config and pipeline 
    ├── utility.py              # Tool and utility functions
    ├── server_setup.py         # Fastapi Server setup
    ├── api_test.ipynb          # Notebook to test fastapi
    └── README.md

## Usage Guide

Follow these steps to use the Spam Check API:

1. Run `model_training.ipynb` to train the machine learning model. The trained model will be saved in the `mlruns` folder.
2. Set up the FastAPI server by running the following command in your command line interface:
   
      ``` uvicorn server_setup:spam_check_api --reload ``` 

   The server will be accessible at http://127.0.0.1:8000/predict

3. Test the API using `api_test.ipynb`. Enter your SMS message into the `message` variable and run the notebook. For example:

   ```message = "Your free ringtone is waiting to be collected. Simply text the password \"MIX\" to 85069 to verify. Get Usher and Britney. FML" ```

   The endpoint will return a classification result (`ham` or `spam`) in JSON format: ```{'prediction': "['spam']"}```


4. Alternatively, you can test the API by visiting http://127.0.0.1:8000/docs.

5. In `model_training.ipynb`, you can customize model pipeline in the **Model Training** section. For example, you can replace tokenization functions, use different vectorizers and add more custom models to experiemt with **PyCaret** (Please refer to https://pycaret.gitbook.io/docs/).

6. For model evaluation and management, run the following command in your command line interface:
```mlflow ui```. Then, navigate to http://127.0.0.1:5000 to manage the registered models. By setting the alias *champion* to the best performing model, it will be deployed to the server once you reload the server.