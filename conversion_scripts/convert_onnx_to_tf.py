import onnx
from onnx_tf.backend import prepare

# Load ONNX model
onnx_model_path = "converted_models/model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX model to TensorFlow SavedModel
tf_model_path = "converted_models/tf_model"
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)

print(f"TensorFlow SavedModel saved to {tf_model_path}")
