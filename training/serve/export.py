# training/serve/export.py
import os
import subprocess
import tensorflow as tf

from training.core.config import get_export_dir, get_input_shape


def export_savedmodel(model, arch: str, out_dir: str = None) -> str:
    """
    Export with a fixed input signature so ONNX conversion uses the correct
    shape instead of inferring from cached metadata.
    """
    if out_dir is None:
        out_dir = str(get_export_dir(arch))
    os.makedirs(out_dir, exist_ok=True)

    input_shape = get_input_shape(arch)

    @tf.function(input_signature=[
        tf.TensorSpec(
            shape=[None, *input_shape],
            dtype=tf.float32,
            name="input_image"
        )
    ])
    def serving_fn(x):
        return model(x, training=False)

    tf.saved_model.save(
        model,
        out_dir,
        signatures={"serving_default": serving_fn}
    )
    print(f"[OK] SavedModel exported to: {out_dir}")
    print(f"     Input signature: [None, {input_shape[0]}, {input_shape[1]}, {input_shape[2]}]")
    return out_dir


def export_to_onnx(saved_model_dir: str, onnx_out_path: str = None) -> str:
    """Converts an exported SavedModel deployment folder cleanly into an ONNX graph structure."""
    if onnx_out_path is None:
        onnx_out_path = f"{saved_model_dir}.onnx"
    try:
        import tf2onnx  # noqa: F401
        cmd = [
            "python", "-m", "tf2onnx.convert",
            "--saved-model", saved_model_dir, 
            "--output",       onnx_out_path,
            "--opset",        "13",
        ]
        subprocess.check_call(cmd)
    except Exception:
        raise RuntimeError("tf2onnx conversion failed or library is not installed in current workspace")
    print(f"[OK] ONNX model exported to: {onnx_out_path}")
    return onnx_out_path

if __name__ == "__main__":
    import argparse
    import keras
    # Import ALL new custom elements required for deserialization
    from training.core.model import SpatialPositionalEmbedding, AttentionPooling, ArcFaceDense
    from training.core.losses import MacroF1, ArcFaceLoss
    from training.training.stages import FastTrainModel
    
    parser = argparse.ArgumentParser(description="Export Emontic AI Model to SavedModel/ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .keras checkpoint file")
    parser.add_argument("--arch", type=str, default="v2s", help="Model architecture (e.g. v2s)")
    parser.add_argument("--out-dir", type=str, default="backend/models/emontic_ai", help="Output directory for the SavedModel")
    
    args = parser.parse_args()
    
    print(f"Loading model safely from {args.checkpoint}...")
    
    # Inject the complete custom blueprint
    loaded_model = keras.models.load_model(
        args.checkpoint, 
        custom_objects={
            "FastTrainModel": FastTrainModel,
            "SpatialPositionalEmbedding": SpatialPositionalEmbedding,
            "AttentionPooling": AttentionPooling,
            "ArcFaceDense": ArcFaceDense,
            "ArcFaceLoss": ArcFaceLoss,
            "MacroF1": MacroF1
        }, 
        compile=False
    )
    
    # ArcFaceDense outputs scaled logits (cos_theta * 30.0). We must apply Softmax
    # during export so the backend ONNX runtime natively receives valid probabilities.
    print("Appending Softmax layer to ArcFace logits...")
    inputs = loaded_model.input
    logits = loaded_model(inputs, training=False)
    probs = keras.layers.Softmax()(logits)
    inference_model = keras.Model(inputs=inputs, outputs=probs)
    
    if os.path.exists(args.out_dir):
        import shutil
        print(f"Removing old model directory: {args.out_dir}")
        shutil.rmtree(args.out_dir)
        
    export_savedmodel(inference_model, args.arch, args.out_dir)
    
    # Convert to ONNX natively
    onnx_path = f"{args.out_dir}.onnx"
    export_to_onnx(args.out_dir, onnx_path)
    
    print(f"Export complete! Deployed ONNX model at: {onnx_path}")