from flask_restful import Resource,reqparse
import tensorflow as tf
import torch
import joblib
import h5py
import json
import dalex as dx
from flask import request
from html2image import Html2Image
from saveinfo import save_file_info
from getmodelfiles import get_model_files


class Importance(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder 
        
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument("params")

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        params=args.get("params")
        params_json= {}
        if params!=None:
            params_json = json.loads(params)
       
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  
        model_task = model_info["model_task"]  
        target_name=model_info["attributes"]["target_names"][0]

        ## loading model
        if backend=="TF1" or backend=="TF2":
            model_h5=h5py.File(model_file, 'w')
            model = tf.keras.models.load_model(model_h5)
        elif backend=="PYT":
            model = torch.load(model_file)
        else:
            model = joblib.load(model_file)
        
        ## loading data
        dataframe = joblib.load(data_file)

        ## params from the request
        kwargsData = dict()
        if "variables" in params_json:
            kwargsData["variables"] = json.loads(params_json["variables"]) if isinstance(params_json["variables"],str) else params_json["variables"]
       
        explainer = dx.Explainer(model, dataframe.drop([target_name], axis=1, inplace=False), dataframe.loc[:, [target_name]],model_type=model_task)
        parts = explainer.model_parts(**{k: v for k, v in kwargsData.items()})

        fig=parts.plot(show=False)
        
        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        fig.write_html(upload_folder+filename+'.html')
        hti = Html2Image()
        hti.output_path= upload_folder
        if "png_height" in params_json and "png_width" in params_json:
            size=(int(params_json["png_width"]),int(params_json["png_height"]))
            hti.screenshot(html_file=upload_folder+filename+'.html', save_as=filename+".png",size=size) 
        else:
            hti.screenshot(html_file=upload_folder+filename+'.html', save_as=filename+".png") 

        response={"plot_html":getcall+'.html',"plot_png":getcall+'.png', "explanation":json.loads(parts.result.to_json())}
        return response


    def get(self):
        return {
        "_method_description": "This method measures the increase in the prediction error of the model after the feature's values are randomly permuted. " 
                                "A feature is considered important if the error of the model increases significantly when permuting it. Accepts 2 arguments: " 
                           "the 'id' string, and the 'params' object (optional) containing the configuration parameters of the explainer."
                           " These arguments are described below.",
        "id": "Identifier of the ML model that was stored locally.",
        "params": { 
                "variables": "(Optional) Array of strings with the names of the features for which the importance will be calculated. Defaults to all features.",
                "png_height": "(optional) height (in pixels) of the png image containing the explanation.",
                "png_width":   "(optional) width (in pixels) of the png image containing the explanation.",
                },
        "output_description":{
                "bar_plot": "A bar plot representing the increase in the prediction error (importance) for the features with the highest values."
               },
        "meta":{
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }
        }
    