from data_utils import read_data, process_compas_data
# from train_utils import *
from small_model import *
from big_model import *
from utils import *
from onnx_utils import *
import tensorflow as tf
from onnx2keras import onnx_to_keras
from alibi.explainers import CounterFactual
from tensorflow import keras
import logging

compas_X, compas_y, compas_actionable_indices, compas_categorical_features, compas_categorical_names = process_compas_data()
compas_experiment_dir = 'results/test_0802_compas/'
compas_data = read_data(compas_experiment_dir)
weight_dir = compas_experiment_dir + '0.0/'
print("loading model...")
model = load_torch_model(weight_dir, "0.0")
print("done")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# k_model = pytorch_to_keras(model, input_var, [(10, 32, 32,)], verbose=True, names='short')  

logger = logging.getLogger('alibi.explainers.counterfactual')
logging.basicConfig(level=logging.CRITICAL)



weight = "0.0"
onnx_model = onnx.load(weight_dir + str(weight) + '_best_model.onnx')


input_all = [node.name for node in onnx_model.graph.input]
input_initializer =  [node.name for node in onnx_model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
tf.compat.v1.disable_eager_execution()


k_model = onnx_to_keras(onnx_model, ['input.1'])

k_model.save(weight_dir + str(weight) + '_best_model.h5')

tf.keras.backend.clear_session()    

new_k_model = keras.models.load_model(weight_dir + str(weight) + '_best_model.h5')


# tf.compat.v1.reset_default_graph()
tf.compat.v1.disable_eager_execution()

new_k_model._make_predict_function()

global graph
graph = tf.compat.v1.get_default_graph()

with graph.as_default():

    sample = compas_data['X_test'].iloc[0].values.reshape(1,-1)

    original_pred = new_k_model.predict(sample).item()
    original_pred = 1 if original_pred > 0.5 else 0
    explainer = CounterFactual(new_k_model, \
                 shape=(1,) + compas_data['X_test'].iloc[0].values.shape, \
                 target_proba = abs(1 - original_pred), target_class='same', \
                 feature_range = (-1e10, 1e10), debug=False)

    recourse = explainer.explain(sample)

    if recourse.cf != None:
        action = (recourse.cf['X'][0]) - sample
        if True:
            print("lambda: ", recourse.cf['lambda'])
            print('index: ', recourse.cf['index'])
            print("action: ", np.around(action, 2))
            print("sample: ", sample)
            print("counterfactual: ", recourse.cf['X'][0])
            print("counterfactual proba: ", recourse.cf['proba'])
            print("normal proba: ", new_k_model.predict(sample))
