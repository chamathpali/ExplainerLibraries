from flask_restful import Resource, reqparse
import json
import pandas as pd
import numpy as np
from getmodelfiles import get_model_files
from saveinfo import save_file_info
import joblib
from html2image import Html2Image
from flask import request
import tensorflow as tf
from tensorflow import keras
import plotly.express as px
 
class iGenCBR(Resource):

    def __init__(self,model_folder,upload_folder):
        self.model_folder = model_folder
        self.upload_folder = upload_folder

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("id", required=True)
        parser.add_argument("instance", required=True)
        parser.add_argument("params")

        #parsing args
        args = parser.parse_args()
        _id = args.get("id")
        params=args.get("params")
        instance = json.loads(args.get("instance"))
        params=args.get("params")
        params_json={}
        if(params!=None):
            params_json = json.loads(params)
        
        #Getting model info, data, and file from local repository
        model_file, model_info_file, data_file = get_model_files(_id,self.model_folder)
       
        # TODO:
        # KERAS FILE .h5  - Craig  
        # DATA FILE .csv  - Craig
        
        # LOAD THE MODEL EXTRACT EMBEDDINGS
        model_embeddings = Sequential()
        model_embeddings.add(LSTM(100, activation = 'relu', return_sequences = True, input_shape = (step_days ,input_train.shape[2])))
        model_embeddings.add(LSTM(32, activation = 'relu', return_sequences=True))
        model_embeddings.add(Dense(1))#, activation='linear'
        model_embeddings.set_weights(model.get_weights()) # <- WE NEED TO CHANGE THIS
        
        # EXTRACTION
        layer_name = 'lstm_3'
        intermediate_layer_model = keras.Model(inputs=model_embeddings.input,
                                               outputs=model_embeddings.get_layer(layer_name).output)
        intermediate_output = intermediate_layer_model(input) # TODO: <- for data
        query_output = intermediate_layer_model(input) # TODO: <- for QUERY
        
        # GET MODEL PREDICTIONS
        predictions = model.predict(input)
        pred_list = []
        for pred in predictions.tolist():
          pred_list.append(pred[0])
        
      # reshaped_output = tf.reshape(
      #             intermediate_output, [len(input), 448]
      # )
      # reshaped_input = numpy.reshape(input, [len(input), 6*14])
      # reshaped_output = reshaped_output.numpy()
      # embeddings = numpy.concatenate((reshaped_output,reshaped_input), axis=1)

        # GETTING THE EXPLANATION 
        input = np.float32(input)
        attributions_ig = []
        for input_x in input:
            attributions_ig.append(integrated_gradients(baseline=np.zeros(input.shape[1:], dtype='float32'),
                                                       input_x=input_x,
                                                       target_class_idx=0,
                                                       m_steps=10))
        attributions_ig = tf.stack(attributions_ig, axis=0, name='stack')
        attributions_ig.shape
        reshaped_input_attributions = numpy.reshape(attributions_ig, [len(attributions_ig), 6*14])        
        
        # KNN OUTPUT RESULTS
        # intermediate_output, query_output - N NEIGHBOURS
        # We get instance -> get the N neighbours from all the training input
        
        # TODO: SCIKIT LEARN KNN CODE         
        
        
        #      {
        #       "score__": 448.0,
        #       "index": "1"
        #     },
        #     {
        #       "score__": 447.47983,
        #       "index": "2"
        #     },
        #     {
        #       "score__": 447.45822,
        #       "index": "2563"
        #     },
        #     {
        #       "score__": 447.43384,
        #       "index": "766"
        #     }
        
        # GENERATE PARALLEL PLOTS

        

        ## loading data
        dataframe = joblib.load(data_file)

        ## params from info
        model_info=json.load(model_info_file)
        backend = model_info["backend"]  ##error handling?
        outcome_name=model_info["attributes"]["target_names"][0]
        feature_names=list(model_info["attributes"]["features"].keys())
        feature_names.remove(outcome_name)
        categorical_features=[]
        for feature in feature_names:
            if isinstance(model_info["attributes"]["features"][feature],list):
                categorical_features.append(dataframe.columns.get_loc(feature))
      
        ## loading model
        # if backend=="TF1" or backend=="TF2":
        #     model=h5py.File(model_file, 'w')
        #     mlp = tf.keras.models.load_model(model)
        #     predic_func=mlp
        # elif backend=="sklearn":
        #     mlp = joblib.load(model_file)
        #     predic_func=mlp.predict_proba
        # elif backend=="PYT":
        #     mlp = torch.load(model_file)
        #     predic_func=mlp.predict
        # else:
        #     mlp = joblib.load(model_file)
        #     predic_func=mlp.predict
        try:
            # if backend=="sklearn":
            mlp = joblib.load(model_file)
            # predic_func=mlp.predict_proba
        except:
            raise "Currently only supports sklearn models"

        

        #getting params from request
        desired_class='opposite'
        feature_attribution_method='LIME'
        attributed_instance='Q'
        if "desired_class" in params_json:
            desired_class = params_json["desired_class"] if params_json["desired_class"]=="opposite" else int(params_json["desired_class"])
        if "feature_attribution_method" in params_json: 
            feature_attribution_method = params_json["feature_attribution_method"]  
        if "attributed_instance" in params_json:
            attributed_instance = params_json["attributed_instance"]

        ## init discern
        discern_obj = discern_tabular.DisCERNTabular(mlp, feature_attribution_method, attributed_instance)    
        dataframe_features = dataframe.drop(dataframe.columns[-1],axis=1).values
        dataframe_labels = dataframe.iloc[:,-1].values
        target_values = dataframe[outcome_name].unique().tolist()
        discern_obj.init_data(dataframe_features, dataframe_labels, feature_names, target_values,**{'cat_feature_indices':categorical_features})

        cf, s, p = discern_obj.find_cf(instance, mlp.predict([instance])[0],desired_class)

        result_df = pd.DataFrame(np.array([instance, cf]), columns=feature_names)

        #saving
        upload_folder, filename, getcall = save_file_info(request.path,self.upload_folder)
        str_html= result_df.to_html()+'<br>'
        
        file = open(upload_folder+filename+".html", "w")
        file.write(str_html)
        file.close()

        hti = Html2Image()
        hti.output_path= upload_folder
        if "png_height" in params_json and "png_width" in params_json:
            size=(int(params_json["png_width"]),int(params_json["png_height"]))
            hti.screenshot(html_str=str_html, save_as=filename+".png",size=size)
        else:
            hti.screenshot(html_str=str_html, save_as=filename+".png")
       
        
        response={"plot_html":getcall+".html","plot_png":getcall+".png","explanation":result_df.to_json()}
        return response

    def get(self):
        return {
        "_method_description": "Discovering Counterfactual Explanations using Relevance Features from Neighbourhoods (DisCERN) generates counterfactuals for scikit-learn-based models. Requires 3 arguments: " 
                           "the 'id' string, the 'instance' to be explained, and the 'params' object containing the configuration parameters of the explainer."
                           " These arguments are described below.",

        "id": "Identifier of the ML model that was stored locally.",
        "instance": "Array of feature values of the instance to be explained.",
        "params": { 
                
                "desired_class": "Integer representing the index of the desired counterfactual class, or 'opposite' in the case of binary classification. Defaults to 'opposite'.",
                "feature_attribution_method": "Feature attribution method used for obtaining feature weights; currently supports LIME, SHAP and Integrated Gradients. Defaults to LIME.",
                "attributed_instance": "Indicate on which instance to use feature attribution: Q for query or N for NUN. Defaults to Q.",
                "png_height": "(optional) height (in pixels) of the png image containing the explanation.",
                "png_width":   "(optional) width (in pixels) of the png image containing the explanation.",
                },

        "output_description":{
                "html_table": "An html page containing a table with the generated couterfactuals."
               },
        "meta":{
                "supportsAPI":False,
                "needsData": True,
                "requiresAttributes":[]
            }
        }

    def compute_gradients(input_xs, target_class_idx):
        with tf.GradientTape() as tape:
            tape.watch(input_xs)
            logits = model(input_xs)
    #         probs = tf.nn.softmax(logits, axis=-1)[:, target_class_idx] # change 2 of 2: commented out as this is a prediction problem
    #     return tape.gradient(probs, input_xs)
        return tape.gradient(logits, input_xs)

    def interpolate_input_xs(baseline,
                           input_x,
                           alphas):
        alphas_x = alphas[:, tf.newaxis, tf.newaxis] # change 1 of 2: Reduced 1 dimension to match the lstm input
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(input_x, axis=0)
        delta = input_x - baseline_x
        input_xs = baseline_x +  alphas_x * delta
        return input_xs

    @tf.function
    def one_batch(baseline, input_x, alpha_batch, target_class_idx):
        # Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_input_xs(baseline=baseline,
                                                           input_x=input_x,
                                                           alphas=alpha_batch)

        # Compute gradients between model outputs and interpolated inputs.
        gradient_batch = compute_gradients(input_xs=interpolated_path_input_batch,
                                           target_class_idx=target_class_idx)
        return gradient_batch

    def integral_approximation(gradients):
        # riemann_trapezoidal
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients

    def integrated_gradients(baseline,
                             input_x,
                             target_class_idx,
                             m_steps=50,
                             batch_size=32):
        # Generate alphas.
        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)

        # Collect gradients.    
        gradient_batches = []

        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for alpha in tf.range(0, len(alphas), batch_size):
            from_ = alpha
            to = tf.minimum(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]

            gradient_batch = one_batch(baseline, input_x, alpha_batch, target_class_idx)
            gradient_batches.append(gradient_batch)

        # Concatenate path gradients together row-wise into single tensor.
        total_gradients = tf.concat(gradient_batches, axis=0)

        # Integral approximation through averaging gradients.
        avg_gradients = integral_approximation(gradients=total_gradients)

        # Scale integrated gradients with respect to input.
        integrated_gradients = (input_x - baseline) * avg_gradients

        return integrated_gradients
