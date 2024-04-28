from sklearn.pipeline import Pipeline
from sklearn import preprocessing

from utility import load_model_config
import mlflow.sklearn

class Modelling():
    """ Modelling: define model pipeline """
    def __init__(self):
        self.model_config = load_model_config()
        self.le = preprocessing.LabelEncoder()
        self.le.fit(self.model_config.encoded_label)

    def load_deployed_model(self) -> object:
        model_name  = self.model_config.deployed_model['model_name']

        try: 
            alias = self.model_config.deployed_model['alias']
            model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
            print (f"Load model {model_name} with alias: {alias}")
        except:
            default_version = self.model_config.deployed_model['model_version']
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{default_version}")
            print (f"Load {model_name} with version: {default_version}")

        return model
    
    def save_model(self, model, artifact_path, model_name, signature=None):
        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        registered_model_name=model_name,
    )

    def make_pipeline(self, vect, model):
        """ make_pipeline: Custom pipeline for model training """
        pipeline = Pipeline([
            ('vect', vect),
            ('clf', model)
        ])
        
        return pipeline
        