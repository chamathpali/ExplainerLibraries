from flask_restful import Resource,reqparse
from flask import request
import tensorflow as tf
import torch
import numpy as np
import joblib
import h5py
import json
import lime.lime_text
from html2image import Html2Image
from saveinfo import save_file_info
from getmodelfiles import get_model_files
import requests

class LimeText(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument('instance',required=True)
        parser.add_argument("url")
        parser.add_argument('params')
        args = parser.parse_args()
        
        _id = args.get("id")
        url = args.get("url")
        instance = args.get("instance")
        params=args.get("params")
        params_json={}
        if(params !=None):
            params_json = json.loads(params)

        
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, _ = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        try:
            output_names=model_info["attributes"]["target_values"][0]
        except:
            output_names=None

        predic_func=None
        if model_file!=None:
            if backend=="TF1" or backend=="TF2":
                model=h5py.File(model_file, 'w')
                mlp = tf.keras.models.load_model(model)
                predic_func=mlp
            elif backend=="sklearn":
                mlp = joblib.load(model_file)
                predic_func=mlp.predict_proba
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
            raise Exception("Either an ID for a locally stored model or an URL for the prediction function of the model must be provided.")
        
     
        # Create explainer
        explainer = lime.lime_text.LimeTextExplainer(class_names=output_names)
        kwargsData2 = dict(labels=(1,), top_labels=None, num_features=None)
        if "output_classes" in params_json:
            kwargsData2["labels"] = json.loads(params_json["output_classes"]) if isinstance(params_json["output_classes"],str) else params_json["output_classes"]  #labels
        if "top_classes" in params_json:
            kwargsData2["top_labels"] = int(params_json["top_classes"])   #top labels
        if "num_features" in params_json:
            kwargsData2["num_features"] = int(params_json["num_features"])

        explanation = explainer.explain_instance(instance, predic_func, **{k: v for k, v in kwargsData2.items() if v is not None}) 
        
        #formatting json explanation
        ret = explanation.as_map()
        ret = {str(k):[(int(i),float(j)) for (i,j) in v] for k,v in ret.items()}
        if output_names!=None:
            ret = {output_names[int(k)]:v for k,v in ret.items()}
        ret=json.loads(json.dumps(ret))

        ##saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        hti = Html2Image()
        hti.output_path= upload_folder
        #size=(1000, 400)
        if "png_height" in params_json and "png_width" in params_json:
            size=(int(params_json["png_width"]),int(params_json["png_height"]))
            hti.screenshot(html_str=explanation.as_html(), save_as=filename+".png",size=size)   
        else:
            hti.screenshot(html_str=explanation.as_html(), save_as=filename+".png")
        explanation.save_to_file(upload_folder+filename+".html")
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":ret}
        return response

    def get(self):
        return {
        "_method_description": "LIME perturbs the input data samples in order to train a simple model that approximates the prediction for the given instance and similar ones. "
                           "The explanation contains the weight of each word to the prediction value. This method accepts 4 arguments: " 
                           "the 'id', the 'instance', the 'url',  and the 'params' JSON with the configuration parameters of the method. "
                           "These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "instance": "A string with the text to be explained.",
        "url": "External URL of the prediction function. Ignored if a model file was uploaded to the server. "
               "This url must be able to handle a POST request receiving a (multi-dimensional) array of N data points as inputs (instances represented as arrays). It must return a array of N outputs (predictions for each instance).",
        "params": { 
                "output_classes" : "(Optional) Array of ints representing the classes to be explained.",
                "top_classes": "(Optional) Int representing the number of classes with the highest prediction probablity to be explained.",
                "num_features": "(Optional) Int representing the maximum number of features to be included in the explanation.",
                "png_height": "(optional) height (in pixels) of the png image containing the explanation.",
                "png_width":   "(optional) width (in pixels) of the png image containing the explanation.",
                },
        "output_description":{
                "lime_plot": "An image contaning a plot with the most influyent words for the given instance. For regression models, the plot displays both positive and negative contributions of each word to the predicted outcome."
                "The same applies to classification models, but there can be a plot for each possible class. The text instance with highlighted words is also included."
               },
        "meta":{
                "supportsAPI":True,
                "needsData": False,
                "requiresAttributes":[]
            }
        }
