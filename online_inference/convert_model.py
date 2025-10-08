import torch
from torchvision.models import resnet18

torch_model = resnet18(pretrained=True)
# Create example inputs for exporting the model. The inputs should be a tuple of tensors.
example_inputs = (torch.randn(1, 3, 224, 224),)
torch_model.eval()
onnx_program = torch.onnx.export(
    torch_model,
    example_inputs,
    input_names=["image"],
    output_names=["label"],
    dynamic_shapes=[{0, "batch_size"}],
    dynamo=True,
)
onnx_program.save("model.onnx")
