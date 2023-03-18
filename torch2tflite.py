import torch
import torch.nn as nn
import torchvision.models as models
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = models.densenet121(pretrained=True)

# Modify the classifier of the model
model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.3),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

state = torch.load('checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(state)
model.eval()
model.to(device)

# Export the PyTorch model to ONNX format
dummy_input = torch.randn(1, 3, 224, 224).to(device)
input_names = ["input"]
output_names = ["output"]
dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
onnx_path = "model.onnx"
torch.onnx.export(model, dummy_input, onnx_path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

# Convert the ONNX model to TensorFlow format
onnx_model = onnx.load(onnx_path)
tf_model = prepare(onnx_model)
tf_model.export_graph("model.pb")

# Convert the TensorFlow model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model("model.pb")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.experimental_new_converter = True
converter.allow_custom_ops = True
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
