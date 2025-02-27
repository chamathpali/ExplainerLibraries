import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import torch
import joblib
import h5py
import json
import shap
import requests
from flask_restful import Resource,reqparse
from flask import request
from saveinfo import save_file_info
from getmodelfiles import get_model_files


class ShapKernelGlobal(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder
        
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('id',required=True)
        parser.add_argument('url')
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)
        
        #getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)


        #getting params from request
        index=1
        if "output_index" in params_json:
            index=int(params_json["output_index"]);

        #getting params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]
        try:
            output_names=model_info["attributes"]["target_values"][0]
        except:
            output_names=None
        target_name=model_info["attributes"]["target_names"][0]
        feature_names=list(model_info["attributes"]["features"].keys())
        feature_names.remove(target_name)
        kwargsData = dict(feature_names=feature_names, output_names=output_names)

        #loading data
        if data_file!=None:
            dataframe = joblib.load(data_file)
            dataframe.drop([target_name], axis=1, inplace=True)
        else:
            raise Exception("The training data file was not provided.")
        #getting predict function
        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend=="sklearn":
                mlp = joblib.load(model_file)
                try:
                    predic_func=mlp.predict_proba
                except:
                    predic_func=mlp.predict
            elif backend=="PYT":
                mlp = torch.load(model_file)
                predic_func=mlp.predict
            else:
                mlp = joblib.load(model_file)
                predic_func=mlp.predict
        elif url!=None:
            def predict(X):
                return np.array(json.loads(requests.post(url, data=dict(inputs=str(X.tolist()))).text))
            predic_func=predict
        else:
            raise Exception("Either a stored model or a valid URL for the prediction function must be provided.")

        #creating explanation
        explainer = shap.KernelExplainer(predic_func, dataframe,**{k: v for k, v in kwargsData.items()})
        shap_values = explainer.shap_values(dataframe)
     
        if(len(np.array(shap_values).shape)==3 and index!=None): #multiclass shape: (#_of_classes, #_of_instances,#_of_features)
            shap_values=shap_values[index]

        #plotting   
        plt.switch_backend('agg')
        shap.summary_plot(shap_values,features=dataframe, feature_names=feature_names,class_names=output_names,show=False)

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        plt.savefig(upload_folder+filename+".png",bbox_inches="tight")

        #formatting json output
        shap_values = [x.tolist() for x in shap_values]
        ret=json.loads(json.dumps(shap_values))
        
        response={"plot_png":getcall+".png","explanation":ret}

        return response


    def get(self):
        return {
        "_method_description": "This method based on Shapley values computes the average contribution of each feature for the whole training dataset. This method accepts 3 arguments: " 
                           "the 'id', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "url": "External URL of the prediction function. Ignored if a model file was previously uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "output_index": "(Optional) Integer representing the index of the class to be explained. Ignore for regression models. Defaults to class 1." 
                },
        "output_description":{
                "beeswarm_plot": "The beeswarm plot is designed to display an information-dense summary of how the top features in a dataset impact the model's output. Each instance the given explanation is represented by a single dot" 
                                 "on each feature fow. The x position of the dot is determined by the SHAP value of that feature, and dots 'pile up' along each feature row to show density. Color is used to display the original value of a feature. "
               },
        "meta":{
                "supportsAPI":True,
                "needsData": True,
                "requiresAttributes":[]
            }
        }
    

