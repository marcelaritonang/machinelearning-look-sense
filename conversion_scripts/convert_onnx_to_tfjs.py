import os
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import tensorflowjs as tfjs

def convert_onnx_to_tfjs(
    onnx_path: str = "converted_models/model.onnx",
    output_dir: str = "converted_models",
    model_name: str = "model"
):
    """
    Mengkonversi model ONNX ke format TFJS
    """
    tf_path = os.path.join(output_dir, f"{model_name}_tf")
    tfjs_path = os.path.join(output_dir, f"{model_name}_tfjs")
    
    # Load dan konversi ONNX
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)
    
    # Konversi ke TFJS
    tfjs.converters.convert_tf_saved_model(
        tf_path,
        tfjs_path
    )
    
    print(f"Model berhasil dikonversi ke: {tfjs_path}")

if __name__ == "__main__":
    convert_onnx_to_tfjs()  